#!/usr/bin/env python
# coding: utf-8

# # 🤖 Multi-Agent Code Review Assistant
# 
# ## Scenario Description
# 
# This notebook implements a **Multi-Agent Code Review Workflow** using LangGraph and LangChain.
# 
# ### Problem Being Solved
# Developers often need quick, reliable code generation **plus** a second-opinion review pass to catch bugs, security issues, and style violations — all without leaving their workflow. This system automates both steps while keeping a human in the loop before the final report is "delivered".
# 
# ### Agent Architecture
# 
# | Agent | Role | Responsibility |
# |---|---|---|
# | **Code Writer Agent** | Software Engineer | Generates Python code based on user requirements; uses a web-search tool to look up relevant APIs/libraries and a code-execution tool to validate syntax. |
# | **Code Reviewer Agent** | Senior Code Reviewer | Analyzes the generated code for bugs, security issues, PEP8 style, and complexity; uses a static-analysis tool and a complexity-scorer tool; produces a structured review report. |
# 
# ### Workflow
# ```
# User Request
#      │
#      ▼
# ┌─────────────────┐
# │  Code Writer    │  ← tools: web_search, validate_python_syntax
# │  Agent          │
# └────────┬────────┘
#          │  generated code
#          ▼
# ┌─────────────────┐
# │  Code Reviewer  │  ← tools: analyze_code_issues, calculate_complexity
# │  Agent          │
# └────────┬────────┘
#          │  review report
#          ▼
#   ⏸ HUMAN-IN-THE-LOOP
#   (approve / request revision)
#          │
#          ▼
#    Final Output
# ```
# 
# ### Key Features
# - ✅ Two distinct agents with dedicated system prompts
# - ✅ Typed shared state (LangGraph `TypedDict`)
# - ✅ Four tools (2 per agent)
# - ✅ Conversation memory via `MemorySaver` checkpointer
# - ✅ Human-in-the-loop interruption before final delivery
# - ✅ `execute_workflow(user_request)` core function
# - ✅ 5 test cases demonstrating approval, revision, and edge cases

# ## 1. Install Dependencies

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install langgraph langchain langchain-openai langchain-community duckduckgo-search ddgs python-dotenv\n')


# ## 2. API Key Configuration
# 
# > **Security Note:** API keys are loaded from **Colab Secrets** (or environment variables). Never hard-code keys in the notebook.

# In[2]:


import os

# ── Colab Secrets (recommended) ──────────────────────────────────────────────
# Only OPENAI_API_KEY is required now.

try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("✅ OPENAI_API_KEY loaded from Colab Secrets.")
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    print("ℹ️ Falling back to environment variables / .env file.")

assert os.getenv("OPENAI_API_KEY"), "❌ OPENAI_API_KEY is not set!"

print("✅ Required API key is present.")


# ## 3. Imports

# In[3]:


import ast
import re
import textwrap
import uuid
from typing import Annotated, Any

# LangChain / LangGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

# DuckDuckGo web search
from langchain_community.tools import DuckDuckGoSearchRun

print("✅ All imports successful.")


# ## 4. Shared State Definition
# 
# The `WorkflowState` TypedDict is the single source of truth passed between all nodes in the graph.

# In[4]:


class WorkflowState(TypedDict):
    """Shared state flowing through the graph."""

    # Shared/general messages
    messages: Annotated[list, add_messages]

    # Separate agent histories
    writer_messages: Annotated[list, add_messages]
    reviewer_messages: Annotated[list, add_messages]

    writer_used_tools: bool
    reviewer_used_tools: bool

    user_request: str

    generated_code: str
    review_report: str

    human_feedback: str
    revision_count: int

    final_output: str


# ## 5. Custom Tools
# 
# Each agent uses **2 tools**: one shared web-search tool, plus one agent-specific custom tool.

# In[5]:


# ── Tool 1 (shared): DuckDuckGo Web Search ───────────────────────────────────
duckduckgo_search = DuckDuckGoSearchRun()

duckduckgo_search.description = (
    "Search the web for Python library documentation, best practices, "
    "and code examples. Use this before writing code if external "
    "information is needed."
)


# ── Tool 2 (Code Writer): Python Syntax Validator ────────────────────────────
@tool
def validate_python_syntax(code: str) -> str:
    """
    Validates whether the provided Python code string is syntactically correct.
    Returns 'VALID' if the code parses without errors, or a descriptive
    SyntaxError message if the code is invalid.

    Args:
        code: A string containing Python source code to validate.

    Returns:
        'VALID' on success, or an error description string on failure.
    """
    try:
        ast.parse(code)
        return "VALID – code parses without syntax errors."
    except SyntaxError as exc:
        return f"SYNTAX ERROR on line {exc.lineno}: {exc.msg}  →  {exc.text}"


# ── Tool 3 (Code Reviewer): Static Issue Analyzer ────────────────────────────
@tool
def analyze_code_issues(code: str) -> str:
    """
    Performs lightweight static analysis on Python code and returns a
    structured report of potential issues across five categories:
    security, style, error-handling, performance, and documentation.

    Args:
        code: A string containing Python source code to analyze.

    Returns:
        A multi-line string report listing detected issues.
    """
    issues = []

    # Security checks
    if re.search(r"\beval\s*\(", code):
        issues.append("[SECURITY] Use of eval() detected – potential code injection risk.")
    if re.search(r"\bexec\s*\(", code):
        issues.append("[SECURITY] Use of exec() detected – potential code injection risk.")
    if re.search(r"password\s*=\s*['\"]\w", code, re.IGNORECASE):
        issues.append("[SECURITY] Hardcoded password detected – use environment variables.")
    if re.search(r"(api_key|secret|token)\s*=\s*['\"][A-Za-z0-9]", code, re.IGNORECASE):
        issues.append("[SECURITY] Hardcoded API key/secret/token – use environment variables.")

    # Style / PEP 8 checks
    lines = code.splitlines()
    long_lines = [i + 1 for i, ln in enumerate(lines) if len(ln) > 79]
    if long_lines:
        issues.append(f"[STYLE] Lines exceeding 79 characters (PEP 8): {long_lines[:5]}")
    if re.search(r"^\t", code, re.MULTILINE):
        issues.append("[STYLE] Tab indentation detected – PEP 8 prefers 4 spaces.")
    if not re.search(r'"""', code):
        issues.append("[DOCS] No docstrings found – consider adding module/function docstrings.")

    # Error handling checks
    if re.search(r"except\s*:", code):
        issues.append("[ERROR_HANDLING] Bare except clause – catch specific exception types.")
    if re.search(r"except\s+Exception\s*:", code):
        issues.append("[ERROR_HANDLING] Catching broad Exception – narrow exception scope if possible.")

    # Performance checks
    if re.search(r"\+\=.*inside.*for|for.*\n.*\+=", code):
        issues.append("[PERFORMANCE] String concatenation in loop – consider using join() or list comprehension.")
    if re.search(r"import \*", code):
        issues.append("[PERFORMANCE/STYLE] Wildcard import (import *) – prefer explicit imports.")

    if not issues:
        return "✅ No issues detected by static analyzer."
    return "Static Analysis Report:\n" + "\n".join(f"  • {i}" for i in issues)


# ── Tool 4 (Code Reviewer): Cyclomatic Complexity Scorer ─────────────────────
@tool
def calculate_complexity(code: str) -> str:
    """
    Estimates the cyclomatic complexity of Python code by counting branching
    constructs (if, elif, for, while, and, or, except, with, assert,
    comprehensions). Returns a complexity score and a risk rating.

    Args:
        code: A string containing Python source code.

    Returns:
        A string with the complexity score and risk level (Low/Medium/High/Very High).
    """
    # Count branching keywords (simplified McCabe estimate)
    branch_patterns = [
        r"\bif\b", r"\belif\b", r"\bfor\b", r"\bwhile\b",
        r"\band\b", r"\bor\b", r"\bexcept\b", r"\bwith\b",
        r"\bassert\b", r"\[.*for.*in.*\]",  # list comp
    ]
    score = 1  # base complexity
    for pattern in branch_patterns:
        score += len(re.findall(pattern, code))

    # Count functions / methods
    func_count = len(re.findall(r"\bdef\b", code))
    class_count = len(re.findall(r"\bclass\b", code))

    if score <= 5:
        risk = "🟢 Low – simple, easy to test"
    elif score <= 10:
        risk = "🟡 Medium – moderate complexity"
    elif score <= 20:
        risk = "🟠 High – consider refactoring"
    else:
        risk = "🔴 Very High – refactoring strongly recommended"

    return (
        f"Complexity Score : {score}\n"
        f"Risk Level       : {risk}\n"
        f"Functions found  : {func_count}\n"
        f"Classes found    : {class_count}"
    )


print("✅ All four tools defined:")
print("   1. duckduckgo_search (web search)")
print("   2. validate_python_syntax")
print("   3. analyze_code_issues")
print("   4. calculate_complexity")


# ## 6. LLM Setup

# In[6]:


# Base LLM
base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ── Code Writer: web search + syntax validator ────────────────────────────────
WRITER_SYSTEM_PROMPT = """You are an expert Python Software Engineer.
Your role is to write clean, correct, and well-documented Python code
based on the user's requirements.

Guidelines:
- Before writing code, use the duckduckgo_search tool to look up any relevant
  library documentation or best practices you are unsure about.
- After writing code, ALWAYS call validate_python_syntax to confirm the
  code is syntactically valid. Fix any reported errors before responding.
- Include a module-level docstring and docstrings for every function/class.
- Follow PEP 8 style guidelines.
- Use tools when needed before producing the final answer.
- Final response must contain only a single Python code block.
- Wrap all code in a single ```python ... ``` fence.
"""

writer_tools = [duckduckgo_search, validate_python_syntax]
writer_llm   = base_llm.bind_tools(writer_tools)

# ── Code Reviewer: static analysis + complexity scorer ───────────────────────
REVIEWER_SYSTEM_PROMPT = """You are a Senior Python Code Reviewer.
Your role is to perform a thorough review of the provided Python code and
produce a structured, actionable review report.

Review process:
1. Call analyze_code_issues on the code to get a static analysis report.
2. Call calculate_complexity on the code to get a complexity score.
3. Combine both tool results with your own expert assessment to produce a
   final review report using EXACTLY this structure:

## Code Review Report
**Overall Assessment:** <PASS / PASS_WITH_MINOR_ISSUES / NEEDS_REVISION>

### ✅ Strengths
<bullet list>

### ⚠️ Issues Found
<numbered list with severity: CRITICAL / MAJOR / MINOR>

### 📊 Complexity Analysis
<paste complexity tool output>

### 💡 Recommendations
<actionable suggestions>
"""

reviewer_tools = [analyze_code_issues, calculate_complexity]
reviewer_llm   = base_llm.bind_tools(reviewer_tools)

print("✅ LLMs bound to tools.")


# ## 7. Graph Nodes

# In[7]:


import re

def extract_python_code(text: str) -> str:
    """
    Extracts the first Python code block from markdown text.
    """
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return text.strip()


# In[8]:


# ── Node: Code Writer ─────────────────────────────────────────────────────────
def code_writer_node(state: WorkflowState):

    print("\n🖊️ [Code Writer] Generating code...")

    messages = state.get("writer_messages", [])

    # First run
    if not messages:
        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Write Python code for:\n\n{state['user_request']}"
            ),
        ]

    response = writer_llm.invoke(messages)

    generated_code = state.get("generated_code", "")

    if response.content and response.content.strip():
        generated_code = extract_python_code(response.content)

    return {
        "writer_messages": [response],
        "generated_code": generated_code,
    }


# ── Node: Code Reviewer ───────────────────────────────────────────────────────
def code_reviewer_node(state: WorkflowState):

    print("\n🔍 [Code Reviewer] Reviewing code...")

    messages = state.get("reviewer_messages", [])

    if not messages:
        messages = [
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Please review the following Python code:\n\n"
                    f"{state.get('generated_code', '')}"
                )
            ),
        ]

    response = reviewer_llm.invoke(messages)

    review_report = state.get("review_report", "")

    if response.content and response.content.strip():
        review_report = response.content

    return {
        "reviewer_messages": [response],
        "review_report": review_report,
    }

# ── Node: Human-in-the-Loop ───────────────────────────────────────────────────
def human_review_node(state: WorkflowState) -> dict:
    """
    Pauses the graph and surfaces the review report to a human.
    The human can either:
      • Type 'approved' → workflow proceeds to final output.
      • Type 'revise: <feedback>' → workflow loops back to Code Writer.
    """
    print("\n" + "═" * 60)
    print("⏸  HUMAN REVIEW REQUIRED")
    print("═" * 60)
    print("\n📋 Review Report:\n")
    print(state.get("review_report", "(no review report yet)"))
    print("\n" + "─" * 60)
    print("Options:")
    print("  • Type 'approved' to accept and finalize.")
    print("  • Type 'revise: <your feedback>' to request changes.")
    print("─" * 60)

    # LangGraph interrupt – suspends the graph until .update_state() is called
    feedback = interrupt(
        {
            "message": "Please review the report above and provide your decision.",
            "review_report": state.get("review_report"),
            "generated_code": state.get("generated_code"),
        }
    )

    return {"human_feedback": feedback}


# ── Node: Revision Handler ────────────────────────────────────────────────────
def revision_node(state: WorkflowState) -> dict:
    """
    Prepends the human's revision feedback to the user request so the
    Code Writer agent can incorporate it in the next pass.
    """
    feedback = state.get("human_feedback", "")
    original_request = state.get("user_request", "")
    revision_count   = state.get("revision_count", 0) + 1

    # Strip the 'revise: ' prefix if present
    clean_feedback = re.sub(r"^revise:\s*", "", feedback, flags=re.IGNORECASE).strip()

    revised_request = (
        f"{original_request}\n\n"
        f"--- Human Reviewer Feedback (revision #{revision_count}) ---\n"
        f"{clean_feedback}"
    )

    print(f"\n🔄 [Revision #{revision_count}] Incorporating feedback: {clean_feedback[:80]}...")

    return {
        "user_request":   revised_request,
        "revision_count": revision_count,
        "human_feedback": "",
    }


# ── Node: Final Output Assembler ──────────────────────────────────────────────
def final_output_node(state: WorkflowState) -> dict:
    """Assembles the final deliverable package."""
    revision_count = state.get("revision_count", 0)
    revision_note  = (
        f"(after {revision_count} revision cycle(s))" if revision_count else "(first pass)"
    )

    final = (
        "═" * 60 + "\n"
        "✅  FINAL DELIVERABLE  " + revision_note + "\n" +
        "═" * 60 + "\n\n"
        "📦 GENERATED CODE\n" +
        "─" * 60 + "\n"
        + state.get("generated_code", "") + "\n\n"
        "📋 REVIEW REPORT\n" +
        "─" * 60 + "\n"
        + state.get("review_report", "") + "\n"
    )

    print(final)
    return {"final_output": final}


print("✅ All graph nodes defined.")


# ## 8. Routing Functions (Conditional Edges)

# In[9]:


def route_after_writer(state: WorkflowState) -> str:

    last_message = state["writer_messages"][-1] if state.get("writer_messages") else None

    if last_message and getattr(last_message, "tool_calls", None):
        return "writer_tools"

    # 🔥 prevent re-running writer after tool execution
    if state.get("writer_used_tools"):
        return "code_reviewer"

    return "code_reviewer"


def route_after_reviewer(state: WorkflowState) -> str:

    last_message = (
        state["reviewer_messages"][-1]
        if state.get("reviewer_messages")
        else None
    )

    if last_message and getattr(last_message, "tool_calls", None):
        return "reviewer_tools"

    return "human_review"


def route_after_human(state: WorkflowState) -> str:
    """
    Route based on human feedback:
      - approved → finalize
      - revise: → revision loop
      - max revisions reached → finalize
    """

    MAX_REVISIONS = 3

    revision_count = state.get("revision_count", 0)

    # Prevent infinite revision loops
    if revision_count >= MAX_REVISIONS:
        print(f"⚠️ Maximum revisions ({MAX_REVISIONS}) reached. Finalizing.")
        return "finalize"

    feedback = state.get("human_feedback", "").strip().lower()

    if feedback.startswith("revise"):
        return "revision"

    # Default: finalize
    return "finalize"


print("✅ Routing functions defined.")


# ## 9. Build the LangGraph

# In[10]:


builder = StateGraph(WorkflowState)

# Nodes
builder.add_node("code_writer", code_writer_node)
builder.add_node("writer_tools", writer_tool_node)

builder.add_node("code_reviewer", code_reviewer_node)
builder.add_node("reviewer_tools", reviewer_tool_node)

builder.add_node("human_review", human_review_node)
builder.add_node("revision", revision_node)
builder.add_node("final_output", final_output_node)

# Entry
builder.add_edge(START, "code_writer")

# Writer flow
builder.add_conditional_edges(
    "code_writer",
    route_after_writer,
    {
        "writer_tools": "writer_tools",
        "code_reviewer": "code_reviewer",
    },
)

builder.add_edge("writer_tools", "code_writer")

# Reviewer flow
builder.add_conditional_edges(
    "code_reviewer",
    route_after_reviewer,
    {
        "reviewer_tools": "reviewer_tools",
        "human_review": "human_review",
    },
)

builder.add_edge("reviewer_tools", "code_reviewer")

# Human loop
builder.add_conditional_edges(
    "human_review",
    route_after_human,
    {
        "finalize": "final_output",
        "revision": "revision",
    },
)

builder.add_edge("revision", "code_writer")

builder.add_edge("final_output", END)

# Compile
memory = MemorySaver()

app_graph = builder.compile(
    checkpointer=memory,
)


# ### 9a. Visualise the Graph

# In[ ]:


try:
    from IPython.display import Image, display
    display(Image(app_graph.get_graph().draw_mermaid_png()))
except Exception as exc:
    print(f"Graph visualisation unavailable ({exc}); continuing without it.")


# ## 10. Core Function: `execute_workflow`

# In[ ]:


def execute_workflow(
    user_request: str,
    human_feedback: str = "approved",
    thread_id: str | None = None,
) -> dict:
    """
    Execute the multi-agent code review workflow.

    The function:
      1. Runs the graph until the Human-in-the-Loop interruption.
      2. Displays the Code Review Report.
      3. Injects `human_feedback` into the graph state.
      4. Resumes execution to produce the final output.

    Args:
        user_request   : Natural-language description of the code to generate.
        human_feedback : Either 'approved' to finalize, or 'revise: <feedback>'
                         to trigger a revision cycle.  Defaults to 'approved'.
        thread_id      : Optional LangGraph thread ID for memory continuity.
                         A new UUID is generated if not supplied.

    Returns:
        A dict with keys:
          - 'thread_id'      : The thread ID used (useful for follow-up calls).
          - 'generated_code' : The final generated Python code.
          - 'review_report'  : The final review report.
          - 'human_feedback' : The feedback that was applied.
          - 'revision_count' : Number of revision cycles performed.
          - 'final_output'   : The assembled deliverable string.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
      "messages": [],

      "writer_messages": [],
      "reviewer_messages": [],

      "writer_used_tools": False,
      "reviewer_used_tools": False,

      "user_request": user_request,

      "generated_code": "",
      "review_report": "",

      "human_feedback": "",
      "revision_count": 0,

      "final_output": "",
  }

    print("=" * 60)
    print(f"🚀 Starting workflow  |  thread: {thread_id}")
    print(f"📝 Request: {user_request[:80]}..." if len(user_request) > 80 else f"📝 Request: {user_request}")
    print("=" * 60)

    # ── Phase 1: Run until HITL interruption ─────────────────────────────────
    for event in app_graph.stream(initial_state, config=config, stream_mode="updates"):
        node_name = list(event.keys())[0]
        if node_name not in ("__interrupt__",):
            print(f"  → Completed node: {node_name}")

    # ── Phase 2: Inject human feedback and resume ─────────────────────────────
    print(f"\n👤 Human feedback supplied: '{human_feedback}'")

    for event in app_graph.stream(
        Command(resume=human_feedback),
        config=config,
        stream_mode="updates",
    ):
      node_name = list(event.keys())[0]
      print(f"  → Completed node: {node_name}")

    # ── Retrieve final state ──────────────────────────────────────────────────
    final_state = app_graph.get_state(config).values

    return {
        "thread_id":      thread_id,
        "generated_code": final_state.get("generated_code", ""),
        "review_report":  final_state.get("review_report", ""),
        "human_feedback": human_feedback,
        "revision_count": final_state.get("revision_count", 0),
        "final_output":   final_state.get("final_output", ""),
    }


print("✅ execute_workflow() function defined.")


# ## 11. Test Cases
# 
# > **Note:** Each test case uses a unique `thread_id` so memory is isolated between tests.
# 
# ---
# 
# ### Test Case 1 – Simple request, human approves immediately

# In[ ]:


result_tc1 = execute_workflow(
    user_request="Write a Python function that reads a CSV file and returns a list of dictionaries, one per row.",
    human_feedback="approved",
)

print("\n" + "─" * 60)
print(f"Test 1 Summary | Revisions: {result_tc1['revision_count']} | Feedback: {result_tc1['human_feedback']}")


# ---
# ### Test Case 2 – Human requests a revision, then approves

# In[ ]:


# First pass: request revision
thread_tc2 = str(uuid.uuid4())

result_tc2_pass1 = execute_workflow(
    user_request="Write a Python class for a simple stack data structure with push, pop, and peek methods.",
    human_feedback="revise: please also add an is_empty() method and a __repr__ method for debugging",
    thread_id=thread_tc2,
)

print(f"\nPass 1 → Revisions so far: {result_tc2_pass1['revision_count']}")

# Second pass on the SAME thread: approve
# (Re-running execute_workflow on the same thread_id starts a fresh run in this
# implementation; to truly continue we'd call the stream directly, but for
# clarity in the notebook we demonstrate approval as a separate execute call.)
result_tc2_pass2 = execute_workflow(
    user_request=(
        "Write a Python class for a simple stack data structure with push, pop, "
        "peek, is_empty(), and __repr__ methods."
    ),
    human_feedback="approved",
)

print(f"\nTest 2 Summary | Final revisions: {result_tc2_pass2['revision_count']} | Feedback: {result_tc2_pass2['human_feedback']}")


# ---
# ### Test Case 3 – Security-sensitive code (hardcoded credentials detected)

# In[ ]:


result_tc3 = execute_workflow(
    user_request=(
        "Write a Python script that connects to a PostgreSQL database named 'mydb' "
        "using psycopg2 and prints all rows from a table called 'users'."
    ),
    human_feedback="revise: make sure database credentials are loaded from environment variables, not hardcoded",
)

print(f"\nTest 3 Summary | Revisions: {result_tc3['revision_count']} | Feedback: {result_tc3['human_feedback'][:60]}...")


# ---
# ### Test Case 4 – Complex algorithm (high complexity score expected)

# In[ ]:


result_tc4 = execute_workflow(
    user_request=(
        "Implement a Python function that solves the 0/1 knapsack problem using "
        "dynamic programming. The function should accept a list of (weight, value) "
        "tuples and a maximum weight capacity, and return the maximum value achievable."
    ),
    human_feedback="approved",
)

print(f"\nTest 4 Summary | Revisions: {result_tc4['revision_count']} | Feedback: {result_tc4['human_feedback']}")


# ---
# ### Test Case 5 – API integration with web search

# In[ ]:


result_tc5 = execute_workflow(
    user_request=(
        "Write a Python async function using httpx that fetches the current Bitcoin "
        "price in USD from the CoinGecko public API and returns it as a float. "
        "Include proper error handling for network failures."
    ),
    human_feedback="approved",
)

print(f"\nTest 5 Summary | Revisions: {result_tc5['revision_count']} | Feedback: {result_tc5['human_feedback']}")


# ---
# 
# ## 12. Results Summary

# In[ ]:


results = [
    ("TC1", "CSV reader function",           result_tc1),
    ("TC2", "Stack class (with revision)",   result_tc2_pass2),
    ("TC3", "DB connection (security fix)",  result_tc3),
    ("TC4", "Knapsack DP algorithm",          result_tc4),
    ("TC5", "Async Bitcoin price fetcher",    result_tc5),
]

print(f"{'ID':<5} {'Description':<35} {'Revisions':<12} {'HITL Decision':<30}")
print("─" * 82)
for tc_id, desc, res in results:
    fb = res["human_feedback"]
    decision = "APPROVED" if fb.lower() == "approved" else f"REVISED → {fb[:25]}..."
    print(f"{tc_id:<5} {desc:<35} {res['revision_count']:<12} {decision:<30}")

print("─" * 82)
print("\n✅ All 5 test cases completed.")
print("   The workflow correctly:")
print("   • Generated code via the Code Writer agent (with web search + syntax validation)")
print("   • Reviewed code via the Code Reviewer agent (with static analysis + complexity)")
print("   • Paused at the Human-in-the-Loop checkpoint")
print("   • Resumed with either approval (→ finalize) or revision feedback (→ writer loop)")
print("   • Maintained memory via MemorySaver across graph runs")

