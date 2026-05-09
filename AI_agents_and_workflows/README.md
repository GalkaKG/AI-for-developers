# AI Research Assistant – Multi-Agent Workflow

**Course:** AI Agents and Workflows for Developers – April 2026  
**Framework:** LangGraph + LangChain  
**Model:** OpenAI GPT-4o-mini

---

## Scenario

An automated research pipeline powered by two AI agents:

1. **Researcher Agent** – searches the web (DuckDuckGo) and Wikipedia to gather information  
2. **Writer Agent** – synthesises findings into a structured report

Before the report is finalised, a **Human-in-the-Loop (HITL)** checkpoint pauses execution and asks the reviewer to approve or request revisions.

---

## Project Files

```
research_assistant_multiagent.ipynb   ← main notebook
README.md                             ← this file
```

---

## Setup

### 1. API Keys

Only one API key is required:

| Key | Where to get it |
|-----|-----------------|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |

DuckDuckGo and Wikipedia require **no API key**.

**In Google Colab:**  
Click the 🔑 key icon in the left sidebar → add a secret named `OPENAI_API_KEY`.

**Locally:**  
```bash
export OPENAI_API_KEY="sk-..."
```

### 2. Install dependencies

The first cell of the notebook runs:

```bash
pip install langchain langchain-openai langchain-community langgraph duckduckgo-search wikipedia
```

---

## How to Run

### Interactive mode
```python
result = execute_workflow("The history of the World Wide Web")
# → agents run, then the graph pauses and prompts:
#   "Your feedback: "
# Type  'approve'  to finalise, or type revision instructions.
```

### Automated / test mode
```python
result = run_workflow_test(
    user_request="Benefits of renewable energy",
    feedback_sequence=["Please add cost statistics.", "approve"]
)
```

---

## Workflow Diagram

```
START
  │
  ▼
researcher_node   (DuckDuckGo + Wikipedia tool-call loop)
  │
  ▼
writer_node       (creates structured report)
  │
  ▼
[INTERRUPT]       ← graph pauses here; human reviews draft
  │
  ▼
human_review_node
  │
  ├─ approved ──► END
  │
  └─ revision ──► writer_node (re-runs with feedback)
                      │
                      ▼
                  [INTERRUPT] ← pauses again for next review
```

State is persisted at every step via `MemorySaver`, so the full conversation history survives across interrupts.

---

## Test Cases

| # | Topic | HITL Scenario |
|---|-------|---------------|
| TC-01 | History of the World Wide Web | Direct approval |
| TC-02 | Benefits of renewable energy | One revision → approval |
| TC-03 | How quantum computing works | Approval via "looks good" |
| TC-04 | AI impact on employment | Two revisions → approval |
| TC-05 | Human microbiome & health | Approval via "accepted" |
