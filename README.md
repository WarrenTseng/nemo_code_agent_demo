# NeMo Code Agent CLI demo

An interactive, simple, multi-agent CLI coding assistant, developed with my AI friends, built for internal corporate networks.
Uses a **Dual-Model Architecture**: a large reasoning model as the **Planner** (via cloud or local NVIDIA NIM and Openai-compatible endpoint)
and a code-specialist model as the **Coder** (also via cloud or local NVIDIA NIM and Openai-compatible endpoint), where the Coder is
registered as a *tool*, not a top-level node, so the Planner drives all coordination.

Built on **LangGraph** + **NeMo Guardrails**, with a `prompt_toolkit` + `Rich` terminal UI, while preserving Nemo Agent Toolkit (NAT) extensibility..

---

## Architecture

```
User Input (prompt_toolkit REPL)
  │
  ▼
Input Guardrails  (politeness check — blocked messages never reach the Planner)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Planner Agent  (Nemotron / large reasoning model)  │
│  LangGraph ReAct loop — orchestrates all tools      │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐   │
│  │ coder_tool  │  │read_file_tool│  │write_file │   │
│  │ (Coder      │  │(filesystem)  │  │  _tool    │   │
│  │  via NIM /  │  └──────────────┘  └───────────┘   │
│  │  vLLM)      │  ┌─────────────────────────────┐   │
│  └─────────────┘  │ execute_bash_tool           │   │
│                   │ (with user confirmation)    │   │
│                   └─────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
  │
  ▼
Output Guardrails  (pattern check + NeMo Guardrails LLM check)
  │
  ▼
Rich-rendered Markdown response
  │
  ▼
SQLite checkpoint  (.agent_logs/checkpoints.db)
```

### Dual-Model LLM Routing

| Role | Default Model | Endpoint |
|------|--------------|----------|
| Planner | `nvidia/nemotron-3-nano-30b-a3b` | NVIDIA NIM (`PLANNER_URL`) |
| Coder | `qwen/qwen2.5-coder-32b-instruct` | NVIDIA NIM (`CODER_URL`) |

Both endpoints are OpenAI-compatible, so any model served via vLLM, Ollama, or NIM works.

---

## Prerequisites

- Python 3.10 or later
- Access to an OpenAI-compatible LLM endpoint for the Planner (Cloud or local NVIDIA NIM and Openai-compatible endpoint)
- Access to a code-specialist LLM endpoint for the Coder (Cloud or local NVIDIA NIM and Openai-compatible endpoint)

---

## Quick Start

```bash
# 1. Clone / navigate to the project
cd nemo_code_agent

# 2. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys and model names

# 3. Run the agent (handles venv creation and install automatically)
./run.sh

# Or with options:
./run.sh --session my-project        # resume a named session
./run.sh --thinking                  # enable Nemotron extended reasoning
./run.sh --checkpoint ./my.db        # custom SQLite checkpoint path
```

`run.sh` automatically installs NAT and NeMo Guardrails on first run and launches via
`nat run` when available, falling back to `code-agent run` otherwise.


---

## Configuration

All configuration is through environment variables (loaded from `.env`):

```bash
# Planner LLM — large reasoning model via NVIDIA NIM
PLANNER_URL=https://integrate.api.nvidia.com/v1 # set to http://your.local-or-cloud.openai-compatible.llm:port/v1, e.g. http://localhost:8000/v1
PLANNER_MODEL=nvidia/nemotron-3-nano-30b-a3b    # 
PLANNER_API_KEY=your_api_key                    # Set to 'none' if your local endpoint does not require authentication.

# Coder LLM — code-specialist model (NIM or local vLLM)
CODER_URL=https://integrate.api.nvidia.com/v1   # set to http://your.local-or-cloud.openai-compatible.llm:port/v1, e.g. http://localhost:8000/v1
CODER_MODEL=qwen/qwen2.5-coder-32b-instruct
CODER_API_KEY=your_api_key                      # Set to 'none' if your local endpoint does not require authentication.
```


---

## CLI Usage

```
code-agent run [OPTIONS]

Options:
  -s, --session TEXT      Session ID — reuse to resume a conversation
  -w, --workspace PATH    Project folder the agent works in
  -c, --checkpoint TEXT   SQLite checkpoint DB path  [default: .agent_logs/checkpoints.db]
  --thinking              Enable extended reasoning mode (Nemotron models)
  --env-file PATH         Custom .env file path
  --history TEXT          prompt_toolkit history file path
```

### Inside the REPL

| Input | Action |
|-------|--------|
| Any text | Send to the Planner agent |
| `exit` / `quit` / `q` / Ctrl-D | Quit |
| Ctrl-C | Interrupt current input (does not quit) |
| ↑ / ↓ arrows | Navigate input history |

---

## How the Agent Works

1. **Input guardrails** check the user message for impolite or unsafe content before it reaches the Planner.
2. **Planner** receives the request and reasons about what steps are needed.
3. To inspect code, it calls `read_file_tool` first.
4. It delegates all code generation to `coder_tool` (the Coder LLM) with a detailed prompt.
5. The Coder returns code blocks; the Planner calls `write_file_tool` to write them to disk.
6. It optionally calls `execute_bash_tool` to run tests or build steps — **you are always asked for approval before any command runs**.
7. The final response passes through output guardrails before display.

---

## Session Persistence

Conversation state is saved to SQLite after every turn via `langgraph-checkpoint-sqlite`.
Resume any previous session by name:

```bash
./run.sh --session my-project-2025
```

---

## Adding Tools

Tools are standard LangChain `@tool` functions. Adding a new tool takes three steps:

### 1. Define the tool

Create a new file in `src/nemo_code_agent/tools/` or add to an existing one:

```python
# src/nemo_code_agent/tools/my_tool.py
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """One-line description — the Planner uses this to decide when to call the tool.

    Args:
        input: describe what this parameter means.

    Returns:
        Result string returned to the Planner.
    """
    # your logic here
    return result
```

### 2. Register the tool in the graph

Open `src/nemo_code_agent/workflow.py` and add your tool to the `tools` list in `build_graph_with_llm`:

```python
from nemo_code_agent.tools.my_tool import my_tool

def build_graph_with_llm(planner_llm, checkpointer):
    tools = [read_file_tool, coder_tool, write_file_tool, execute_bash_tool, my_tool]
    ...
```

### 3. Update the Planner system prompt (optional)

Add a row to the tools table in `PLANNER_SYSTEM_PROMPT` in `workflow.py` so the Planner knows when to use it:

```python
| `my_tool` | Brief description of when the Planner should call it. |
```

That's it — the Planner will automatically discover the tool schema and call it when appropriate.

---

## Debug Logging

All inter-agent payloads, guardrail events, and tool I/O are written to:
```
.agent_logs/debug.log
```
The terminal UI shows only user-facing output. To tail the log in another terminal:
```bash
tail -f .agent_logs/debug.log
```
