# Analytics Agent (LangGraph + Gemini)

A lightweight, multi‑agent workflow that turns a plain‑English analytics request into a plotted chart.  
It uses **LangGraph** for orchestration, **Gemini (Google Generative AI)** for reasoning/codegen, and **pandas + matplotlib** for data handling and visualization.

---

## 1. What it does

Given a user query like _“plot daily sales trend for July”_, the agent:
1. **Plans** the next step based on current state.
2. **Detects data**: selects the best table from a catalog (`data/schema.json`).
3. **Creates a plot**: asks the LLM to generate safe, runnable **matplotlib** code against the selected CSV and saves the figure to `data/charts/`.
4. **Finalizes**: returns the state (including the chart path) and logs visible **events** for progress tracking.

---

## 2. Architecture

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

## 3. Project structure

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

## 4. Configuration

Create a `.env` file in the project root with your Google API key:

```env
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

> The code loads this from `python-dotenv` automatically.

---

## 5. Data Setup (`data/schema.json`)

A simple JSON dictionary mapping **table keys** to metadata:

```json
{
  
  "inventory": {
    "description": "Ending inventory by SKU and date.",
    "path": "data/inventory.csv",
    "schema": {
      "date": "invoice date",
      "sku": "product sku code",
      "stock": "# of units in 1000's"
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

## 6. Installation

Create a virtual environment (any tool is fine). Two options are shown:

### pip + venv
```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
---

## 7. Usage

### 1) Prepare data
- Put your CSV files under `data/`.
- Describe them in `data/schema.json` as shown above.

### 2) Run from Python
```python
from analytics_agent import run_analytics

# Any business question that can be answered from your catalog
result = run_analytics("Plot total sales by region for July 2024")

print("Selected table:", result.get("table_name"))
print("Chart path    :", result.get("image_path")) # chart stored here
print("\nEvents:")
for e in result.get("events", []):
    print(f"[{e['ts']}] {e['agent']}: {e['message']} (p={e['progress']})")
```

### 3) Output
- A PNG chart is saved under `data/charts/` (path also returned in `state["image_path"]`).

---


## 8. Extending the workflow

Scaffolds exist for future nodes:
- **ask_clarifying_question**: when the query is ambiguous.
- **reflexion_code**: auto‑improve chart aesthetics/semantics.
- **code_fix**: catch and repair LLM‑generated code errors, then retry.
- **expose_web_search_api**: To fetch data from external source (Web) 

---
