# multi_agent_langgraph.py
from __future__ import annotations
import os, json, re, uuid, traceback
from datetime import datetime
from typing import TypedDict, Optional, List, Dict, Any

import pandas as pd
from PIL import Image

# LangChain / LangGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

SCHEMA_PATH = "data\\schema.json"
CHART_DIR   = "data\\charts"
os.makedirs(CHART_DIR, exist_ok=True)

with open(SCHEMA_PATH, "r") as f:
    SCHEMA: Dict[str, Any] = json.load(f)

# ----------------------------
# 1) STATE
# ----------------------------
class AState(TypedDict, total=False):
    # input
    query: str

    # clarification_states
    needs_clarification: bool
    clarification_question: str
    clarification_answer: str

    # code fix step
    last_code: Optional[str]
    last_error: Optional[str]
    retry_count: int
    max_retries: int

    # detection outputs
    table_name: Optional[str]
    file_path: Optional[str]
    table_schema: Optional[Dict[str, Any]]
    next_step: Optional[str]

    # plotting outputs
    image_path: Optional[str]

    # events (for visibility)
    events: List[Dict[str, Any]]

def _event(state: AState, agent: str, message: str, progress: float, **kw):
    state.setdefault("events", []).append({
        "id": str(uuid.uuid4())[:8],
        "agent": agent,
        "message": message,
        "progress": progress,
        "ts": datetime.now().isoformat(timespec="seconds"),
        **kw
    })
    e = state["events"][-1]
    print(f" - [{e['ts']}] {e['agent']}: {e['message']} (p={e['progress']})")


# ----------------------------
# 2) SHARED HELPERS (ported)
# ----------------------------
def fetch_table_description() -> str:
    tdescription = ""
    for tname in SCHEMA:
        tdescription += f"{tname} : {SCHEMA[tname]['description']}\n"
    return tdescription

def _clean_exe_code(text: str) -> str:
    text = text.strip()
    if "<execute_python>" not in text:
        text = f"<execute_python>\n{text}\n</execute_python>"
    m = re.search(r"<execute_python>([\s\S]*?)</execute_python>", text)
    if not m:
        raise ValueError("No <execute_python> code block found.")
    code = m.group(1).strip()
    return code

# ----------------------------
# 3) AGENTS (nodes)
# ----------------------------
def planner_orchestrator(state: AState) -> AState:
    """Central desicion maker to call different agents"""
    # _event(state, "Planner/Orchestrator", "Evaluating workflow state", 0.1)
    if( not state.get("table_name")):
        next_step = "detect"
        msg = "Data not selected. Routing to data_detection()"
        completeness = 0.1
    elif(not state.get("image_path")):
        next_step = "plot"
        msg = "Data is selected but charts not prepared. Routing to plot_creation()"
        completeness = 0.3
    elif state.get("next_step") != END:
        next_step = "finalize"
        msg = "Data selected and chart prepared. Routing to finalize()."
        completeness = 0.9
    else:
        next_step = END
        msg = "Workflow completed"
        completeness = 1
    state["next_step"] = next_step
    _event(state, "Planner/Orchestrator", msg, completeness, next=next_step)
    return state  # nothing to mutate yet

def data_detection(state: AState) -> AState:
    query = state["query"]
    system_prompt = f"""
You are a table router system that detects which data source to select based on user's query.
Return ONLY the table name (exact key) from the provided catalog. If none fits, return "NONE".
Tables (name : description):
{fetch_table_description()}
"""
    human_prompt = f"Find the associated table for the following query:\n{query}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    resp = llm.invoke(messages)  # -> AIMessage
    candidate = (resp.content or getattr(resp, "text", "")).strip()

    if candidate in SCHEMA:
        state["table_name"] = candidate
        state["file_path"] = SCHEMA[candidate]["path"]
        state["table_schema"] = SCHEMA[candidate]["schema"]
        _event(state, "DataDetection", f"Selected table: {candidate}", 0.3, table=candidate)
    else:
        state["table_name"] = None
        state["file_path"] = None
        state["table_schema"] = None
        _event(state, "DataDetection", f"No suitable table for query. Model said: {candidate}", 1.0, level="warn")
        raise ValueError(f"DataDetection failed: LLM returned '{candidate}'")

    return state
    
def plot_creation(state: AState) -> AState:
    """Generates executable matplotlib code from df + query; saves figure and closes."""
    if not state.get("file_path") or not state.get("table_schema"):
        raise ValueError("PlotCreation requires file_path and table_schema in state.")

    # image path
    img_name = f"chart_{state['table_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"
    image_path = os.path.join(CHART_DIR, img_name)
    state["image_path"] = image_path

    # prompt
    system_prompt = f"""
You are a data visualization experlot. Create a matptlib plot from a pandas DataFrame named 'df'.
You may transform data as needed. Return ONLY executable Python code inside these tags:

<execute_python>
# valid python code here
</execute_python>

Requirements:
1) Assume data is loaded in 'df' (a pandas DataFrame).
2) Use matplotlib only for plotting (no seaborn).
3) Add clear title, axis labels, and legend if needed.
4) Save the figure as '{image_path}' with dpi=300.
5) Do NOT call plt.show().
6) Close all figures with plt.close() if opened any.
7) Include all necessary imports.
"""
    human_prompt = f"""Generate code for this schema and query:
table_schema:
{json.dumps(state['table_schema'], indent=2)}
query:
{state['query']}
"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    resp = llm.invoke(messages)
    code = _clean_exe_code(resp.text)
    state["last_code"] = code

    # Execute with df loaded
    _event(state, "PlotCreation", "Generating chart code", 0.5)
    df = pd.read_csv(state["file_path"], encoding="latin1")
    exec_env: Dict[str, Any] = {"df": df}
    # try:
    exec(code, exec_env)
    state["last_error"] = None
    # except Exception as e:
    #     code_error = traceback.format_exc()
    #     state["last_error"] = code_error
    #     state["retry_count"] = state.get("retry_count", 0) + 1

    # Image.open(image_path).show()

    _event(state, "PlotCreation", f"Chart saved: {image_path}", 0.9, artifact=image_path)
    return state

def finalize(state: AState) -> AState:
    if not state.get("image_path") or not os.path.exists(state["image_path"]):
        raise FileNotFoundError("Finalize: chart image was not created.")
    _event(state, "Orchestrator", "Workflow complete", 1.0)
    state["next_step"] = END
    return state

def ask_clarifying_question(state: AState) -> AState:
    """Take second input in case of ambigous statement"""
    ## intermediate node at planner step
    ## --todo--

def reflexion_code(state: AState) -> AState:
    """improve generated charts in case of chart quality is not good"""
    ## node after plot_creation node
    ## --todo--

def code_fix(state: AState) -> AState:
    """Rectifies the error from the llm generated python code """
    ## conditional edge to this node when plot_creation fails
    ## -- todo --

# ----------------------------
# 4) BUILD GRAPH
# ----------------------------
def build_graph():
    g = StateGraph(AState)

    # nodes
    g.add_node("plan", planner_orchestrator)
    g.add_node("detect", data_detection)
    g.add_node("plot", plot_creation)
    g.add_node("finalize", finalize)

    g.set_entry_point("plan")

    # planner routes based on state["next_step"]
    def plan_router(state: AState):
        return state.get("next_step", END)

    g.add_conditional_edges(
        "plan",
        plan_router,
        {
            "detect": "detect",
            "plot": "plot",
            "finalize": "finalize",
            END: END,
        },
    )

    # after each agent, return to planner (cyclic)
    g.add_edge("detect", "plan")
    g.add_edge("plot", "plan")
    g.add_edge("finalize", "plan")

    memory = MemorySaver()
    app = g.compile(checkpointer=memory)
    return app

# ----------------------------
# 5) RUNNER
# ----------------------------
def run_analytics(query: str) -> AState:
    app = build_graph()
    print(app)
    initial: AState = {"query": query}
    # thread_id lets you resume or inspect; set any stable id if you want
    result: AState = app.invoke(initial, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    return result

