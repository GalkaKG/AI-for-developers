# Multi-Agent Code Review Assistant
### AI Agents and Workflows for Developers – April 2026 Individual Assignment

---

## 📖 Overview

A **stateful multi-agent workflow** built with **LangGraph + LangChain** that:
1. Generates Python code from a natural-language request (Code Writer Agent)
2. Reviews the generated code for bugs, security issues, style, and complexity (Code Reviewer Agent)
3. Pauses for a **Human-in-the-Loop** decision (approve or request revision)
4. Either delivers the final result or loops back for another generation cycle

---

## 🏗 Architecture

```
User Request
     │
     ▼
┌─────────────────┐
│  Code Writer    │  ← tools: tavily_search, validate_python_syntax
│  Agent          │
└────────┬────────┘
         │  generated code
         ▼
┌─────────────────┐
│  Code Reviewer  │  ← tools: analyze_code_issues, calculate_complexity
│  Agent          │
└────────┬────────┘
         │  review report
         ▼
  ⏸ HUMAN-IN-THE-LOOP interrupt
  (approve / request revision)
         │
    ┌────┴────┐
    ▼         ▼
 Finalize  Revision → back to Code Writer
```

---

## 🔧 Requirements

### API Keys (Colab Secrets or environment variables)

| Key | Source | Required? |
|-----|--------|-----------|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | ✅ Yes |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) | ✅ Yes |

### Colab Setup

In Google Colab, add secrets via **🔑 Key icon (left sidebar) → Add new secret**:
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

The first notebook cell installs all Python packages automatically:
```
langgraph langchain langchain-openai langchain-community tavily-python python-dotenv
```

---

## 🚀 Usage

### Core function signature

```python
execute_workflow(
    user_request: str,          # What code to generate
    human_feedback: str = "approved",  # "approved" OR "revise: <feedback>"
    thread_id: str | None = None       # Optional for memory continuity
) -> dict
```

### Example – approve immediately

```python
result = execute_workflow(
    user_request="Write a function that validates email addresses using regex.",
    human_feedback="approved"
)
print(result["final_output"])
```

### Example – request a revision

```python
result = execute_workflow(
    user_request="Write a Python stack class.",
    human_feedback="revise: please add an __iter__ method and type hints"
)
```

---

## 📋 Agent Details

### Code Writer Agent
- **System prompt:** Senior Python Software Engineer
- **Tools used:**
  - `tavily_search` – looks up library docs/best practices before coding
  - `validate_python_syntax` – validates generated code with `ast.parse`

### Code Reviewer Agent
- **System prompt:** Senior Code Reviewer
- **Tools used:**
  - `analyze_code_issues` – lightweight static analyzer (security, style, error-handling, performance, docs)
  - `calculate_complexity` – cyclomatic complexity estimator with risk rating

---

## 🧪 Test Cases

| # | Request | HITL Decision |
|---|---------|---------------|
| TC1 | CSV reader function | Approved ✅ |
| TC2 | Stack class | Revised (add methods) → Approved ✅ |
| TC3 | PostgreSQL connection | Revised (fix hardcoded creds) ✅ |
| TC4 | 0/1 Knapsack DP | Approved ✅ |
| TC5 | Async Bitcoin price API | Approved ✅ |

---

## 📁 File Structure

```
submission.zip
├── code_review_multiagent.ipynb   # Main notebook
└── README.md                      # This file
```

---

## ⚙️ Memory & HITL Implementation

- **Memory:** `MemorySaver` checkpointer passed to `builder.compile()` — persists full graph state including message history across `stream()` calls within the same `thread_id`.
- **HITL:** Graph compiled with `interrupt_before=["human_review"]`. The `execute_workflow` function:
  1. Streams the graph until it pauses at the `human_review` node
  2. Calls `app_graph.update_state()` to inject the human's decision
  3. Resumes execution with `app_graph.stream(None, config=config)`

---

*Built with LangGraph 0.2+, LangChain 0.3+, and OpenAI GPT-4o-mini*
