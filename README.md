# An example for Code Agent CLI using NeMo

An interactive, simple, multi-agent CLI coding assistant, developed with my AI friends, built for internal corporate networks.
Uses a **Dual-Model Architecture**: a large reasoning model as the **Planner** (via cloud or local NVIDIA NIM and Openai-compatible endpoint)
and a code-specialist model as the **Coder** (also via cloud or local NVIDIA NIM and Openai-compatible endpoint), where the Coder is
registered as a *tool*, not a top-level node, so the Planner drives all coordination.

Built on **LangGraph** + **NeMo Guardrails** + **NeMo Agent Toolkit (NAT)**, with a `prompt_toolkit` + `Rich` terminal UI.

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

This architecture is completely model-agnostic. Since it uses standard OpenAI-compatible endpoints, you can easily swap out the default models for any API of your choice, including local deployments via **vLLM**, **Ollama**, or on-premise **NIM**.

For demonstration purposes, the default configuration below uses a smaller, lightweight model (`nvidia/nemotron-3-nano-30b-a3b`) for both the Planner and Coder roles via the [NVIDIA NIM](https://build.nvidia.com/) trial endpoints. Depending on your specific needs and task complexity, you can swap these out for more powerful models. For example, you might use a large reasoning model like `nvidia/nemotron-3-super-120b-a12b` for the Planner and a specialized model like `qwen/qwen3-coder-480b-a35b-instruct` for the Coder.

If you'd like to try out the NIM endpoints, please visit their website to register and obtain a trial API key.

| Role | Default Model | Endpoint |
|------|--------------|----------|
| Planner | `nvidia/nemotron-3-nano-30b-a3b` | NVIDIA NIM (`PLANNER_URL`) |
| Coder | `nvidia/nemotron-3-nano-30b-a3b` | NVIDIA NIM (`CODER_URL`) |

---

## Prerequisites

- Python 3.10 or later, 3.12 is recommended
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
PLANNER_MODEL=nvidia/nemotron-3-super-120b-a12b    
PLANNER_API_KEY=your_api_key                    # Set to 'none' if your local endpoint does not require authentication. Or get an API key here: https://build.nvidia.com/

# Coder LLM — code-specialist model (NIM or local vLLM)
CODER_URL=https://integrate.api.nvidia.com/v1   # set to http://your.local-or-cloud.openai-compatible.llm:port/v1, e.g. http://localhost:8000/v1
CODER_MODEL=qwen/qwen3-coder-480b-a35b-instruct      # qwen/qwen2.5-coder-32b-instruct
CODER_API_KEY=your_api_key                      # Set to 'none' if your local endpoint does not require authentication. Or get an API key here: https://build.nvidia.com/
```
For more details, please refer to `.env.example`.

---

## CLI Usage

```
code-agent run [OPTIONS]

Options:
  -s, --session TEXT      Session ID — reuse to resume a conversation
  -w, --workspace PATH    Project folder the agent works in
  -c, --checkpoint TEXT   SQLite checkpoint DB path  [default: .agent_logs/checkpoints.db]
  --thinking              Enable extended reasoning mode (Nemotron models)
  --show-thinking         Stream the Planner's reasoning tokens live in the terminal
  --auto-approve          Skip approval prompts for bash commands and file writes (sandbox/Docker only)
  --env-file PATH         Custom .env file path
  --history TEXT          prompt_toolkit history file path
```

### Inside the REPL

| Input | Action |
|-------|--------|
| Any text | Send to the Planner agent |
| `/thinking` | Expand the last buffered reasoning trace |
| `exit` / `quit` / `q` / Ctrl-D | Quit |
| ESC | Interrupt the agent mid-stream |
| Ctrl-C | Interrupt current input (does not quit) |
| ↑ / ↓ arrows | Navigate input history |
| Tab | Autocomplete from history |

---

## Features

### Dual-Mode Coding: Planner-only vs Planner + Coder

Control whether a specialist Coder LLM is used for code generation:

```bash
ENABLE_CODER=true   # (default) Planner delegates all code generation to coder_tool
ENABLE_CODER=false  # Planner writes code directly using its own reasoning
```

When `ENABLE_CODER=true`, the Planner **never writes code itself** — it always calls `coder_tool`, then `write_file_tool`. When disabled, the Planner handles everything in one model.

---

### Knowledge Base (Static Rules + RAG)

Inject project-specific rules and documentation into every code generation call.

```
knowledge/
    static.md       ← always injected (Feature 1)
    RAG/
        *.md        ← retrieved by semantic search (Feature 2)
```

- **Strategy 1 — Static rules**: Place guidelines in `knowledge/static.md`. Every code generation call will include these as mandatory rules (e.g. naming conventions, style rules, architecture patterns).
- **Strategy 2 — RAG retrieval**: Place reference `.md` files in `knowledge/RAG/`. When enabled, the most relevant chunks are retrieved and injected based on the current task.

Knowledge is routed automatically based on mode:

| `ENABLE_CODER` | Where knowledge is injected |
|---|---|
| `true` | into `coder_tool` at generation time |
| `false` | into the Planner context each turn |

```bash
ENABLE_KNOWLEDGE=true                                   # master on/off switch (default: true)
CODER_RAG_ENABLED=false                                 # enable RAG retrieval (default: false)
CODER_RAG_STORE=chroma                                  # define to save the embeddings in disk or memory
CODER_KNOWLEDGE_DIR=knowledge                           # path to knowledge directory
CODER_EMBEDDING_URL=...                                 # openai-compatible embedding endpoint
CODER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-1b-v2 # an example using NIM endpoint
CODER_EMBEDDING_INPUT_TYPE=asymmetric                   # symmetric or asymmetric, depends on the type of embedding model
```

---

### Thinking / Extended Reasoning

For models that support extended reasoning (e.g. Nemotron):

```bash
./run.sh --thinking              # enable extended reasoning mode
./run.sh --thinking --show-thinking  # stream reasoning tokens live in terminal
```

When `--show-thinking` is off (default), reasoning is buffered silently. Type `/thinking` in the REPL to expand the last turn's reasoning trace.

---

### ⚠️ Auto-Approve Mode

By default, the agent asks for your approval before every bash command and file write. For fully automated use in sandboxed environments:

```bash
./run.sh --auto-approve
```

A security warning is shown and you must type `YES I UNDERSTAND` to confirm. Recommended only in Docker containers, sandbox VMs, or CI/CD pipelines.

---

## How the Agent Works

1. **Input guardrails** check the user message for impolite or unsafe content before it reaches the Planner.
2. **Planner** receives the request and reasons about what steps are needed.
3. To inspect code, it calls `read_file_tool` first.
4. If `ENABLE_CODER=true`: delegates code generation to `coder_tool` (Coder LLM), which receives static rules + RAG context. If `ENABLE_CODER=false`: Planner writes code directly, with knowledge injected into its own context.
5. The Planner calls `write_file_tool` to write code to disk — **you are asked for approval**.
6. It optionally calls `execute_bash_tool` to run tests or build steps — **you are also asked for approval**.
7. The final response passes through output guardrails before display.

---

## Session Persistence

Conversation state is saved to SQLite after every turn via `langgraph-checkpoint-sqlite`.
Resume any previous session by name:

```bash
./run.sh --session my-project-2025
```

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

