# Analytics Agent (LangGraph + Gemini)

A lightweight, multi‑agent workflow that turns a plain‑English analytics request into a plotted chart.  
It uses **LangGraph** for orchestration, **Gemini (Google Generative AI)** for reasoning/codegen, and **pandas + matplotlib** for data handling and visualization.

---

## ✨ What it does

Given a user query like _“plot daily sales trend for July”_, the agent:
1. **Plans** the next step based on current state.
2. **Detects data**: selects the best table from a catalog (`data/schema.json`).
3. **Creates a plot**: asks the LLM to generate safe, runnable **matplotlib** code against the selected CSV and saves the figure to `data/charts/`.
4. **Finalizes**: returns the state (including the chart path) and logs visible **events** for progress tracking.

---

## 🧩 Architecture

The workflow is implemented with **LangGraph** and a shared state `AState`:

- **plan (Planner/Orchestrator)**  
  Routes to the next node: `detect` → `plot` → `finalize` based on what’s already done.

- **detect (DataDetection)**  
  Uses Gemini to map the natural‑language query to a table **name** from `schema.json`.  
  Sets `table_name`, `file_path`, and `table_schema` in the state.

- **plot (PlotCreation)**  
  Prompts Gemini to produce **executable Python** that:
  - Assumes a pandas `DataFrame` named `df` (loaded from the selected CSV).
  - Uses **matplotlib only** (no seaborn).
  - Adds title/labels/legend when useful.
  - **Saves** to a file path inside `data/charts/` and **does not show** the plot.
  - **Closes** figures to avoid resource leaks.

- **finalize**  
  Verifies the chart exists, marks workflow complete.

The graph is cyclic: after each node, control returns to **plan** until `END`.

---

## 🗂️ Project structure

```
.
├─ analytics_agent.py            # Main module (builds graph + runner)
├─ data/
│  ├─ schema.json                # Table catalog (see format below)
│  └─ charts/                    # Auto‑created; generated plots are saved here
├─ requirements.txt
└─ .env                          # Holds GOOGLE_API_KEY
```

---

## 🔑 Configuration

Create a `.env` file in the project root with your Google API key:

```env
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

> The code loads this from `python-dotenv` automatically.

---

## 🧱 Data catalog (`data/schema.json`)

A simple JSON dictionary mapping **table keys** to metadata:

```json
{
  "sales_daily": {
    "description": "Daily sales aggregated by date and region.",
    "path": "data/sales_daily.csv",
    "schema": {
      "date": "YYYY-MM-DD",
      "region": "string",
      "sales": "number"
    }
  },
  "inventory": {
    "description": "Ending inventory by SKU and date.",
    "path": "data/inventory.csv",
    "schema": {
      "date": "YYYY-MM-DD",
      "sku": "string",
      "stock": "number"
    }
  }
}
```

- **Key** (e.g., `sales_daily`) is what the LLM must return to select a table.
- **description** is crucial—keep it clear so the model can route correctly.
- **path** must point to a readable CSV.
- **schema** is a human‑readable field dictionary the plotter includes in its prompt.

> The agent reads `SCHEMA_PATH = "data\\schema.json"` on Windows‑style paths. On macOS/Linux you can use `data/schema.json` as well (Python will open it fine if the file exists at that path).

---

## 📦 Installation

Create a virtual environment (any tool is fine). Two options are shown:

### Option A) `pip` + `venv`
```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B) `uv` (fast Python package manager)
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` (included):
```txt
langgraph
langchain
langchain_google_genai
pandas
matplotlib
python-dotenv
```

> Tested with Python 3.11–3.12. If you pin versions, ensure `langchain_google_genai` matches the expected `google-ai-generativelanguage` backend for your environment.

---

## ▶️ Usage

### 1) Prepare data
- Put your CSV files under `data/`.
- Describe them in `data/schema.json` as shown above.

### 2) Run from Python
```python
from analytics_agent import run_analytics

# Any business question that can be answered from your catalog
result = run_analytics("Plot total sales by region for July 2024")

print("Selected table:", result.get("table_name"))
print("Chart path    :", result.get("image_path"))
print("\nEvents:")
for e in result.get("events", []):
    print(f"[{e['ts']}] {e['agent']}: {e['message']} (p={e['progress']})")
```

### 3) Output
- A PNG chart is saved under `data/charts/` (path also returned in `state["image_path"]`).

---

## 🧪 How code execution works

- The **LLM returns code** wrapped inside `<execute_python> ... </execute_python>` tags.
- The agent extracts that snippet, injects an execution environment with `df` already loaded (`pd.read_csv(file_path, encoding="latin1")`), and runs it.
- The snippet **must** save the image to the provided path, and **must** not call `plt.show()`.

A minimal successful snippet (the LLM is guided to produce this style):
```python
import matplotlib.pyplot as plt

# transform/aggregate df as needed...
df_grouped = df.groupby("date", as_index=False)["sales"].sum()

plt.figure()
plt.plot(df_grouped["date"], df_grouped["sales"])
plt.title("Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(r"data/charts/chart_sales_daily.png", dpi=300)
plt.close()
```

---

## 🚦 Events & visibility

Each node appends an **event** with timestamp, agent name, message, and progress (0–1).  
You can render these to show planner decisions, detected table, and where the chart was saved.

---

## 🧰 Troubleshooting

- **“DataDetection failed”**: The LLM couldn’t map your query to a key in `schema.json`.  
  - Improve `description` fields in the catalog.  
  - Try simpler phrasing in your query.  
  - Ensure the key it returned actually exists.

- **CSV encoding issues**: The runner uses `encoding="latin1"`. If your CSVs are UTF‑8 only, update the loader accordingly.

- **Empty/incorrect plots**: Enhance `schema` details and table descriptions; the model relies on them to infer columns and aggregations.

- **Charts not saved**: The finalize step checks for the file. Ensure the LLM code uses the **exact** `image_path` provided in the prompt.

---

## 🧱 Extending the workflow

Scaffolds exist for future nodes:
- **ask_clarifying_question**: when the query is ambiguous.
- **reflexion_code**: auto‑improve chart aesthetics/semantics.
- **code_fix**: catch and repair LLM‑generated code errors, then retry.

To add a new node:
1. Implement a function `(state) -> state`.
2. `g.add_node("name", fn)` and add edges/conditional routes.
3. Update the planner’s routing logic if needed.

---
