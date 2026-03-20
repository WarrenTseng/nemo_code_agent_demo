"""NAT-compatible LangGraph export.

Exposes the compiled code-agent graph as a module-level variable so
NAT's ``langgraph_wrapper`` workflow type can load it directly:

    workflow:
      _type: langgraph_wrapper
      graph: src/nemo_code_agent/nat_graph.py:graph
      env: .env

Uses ``MemorySaver`` as the checkpointer since NAT manages sessions
externally via its own state/thread tracking.  For persistent sessions
use the interactive REPL (``code-agent run``) instead.

⚠ LLM sourcing note:
    Because the graph is compiled once at module-import time, the
    ``langgraph_wrapper`` path reads LLMs directly from environment
    variables (PLANNER_URL, PLANNER_MODEL, etc.) loaded from ``.env``.
    The ``planner_llm`` / ``coder_llm`` entries in ``config.yml`` are
    **not used** by this path — they are only used when NAT Builder
    provisions the LLM explicitly (e.g. the interactive REPL path via
    ``create_agent()``, or the registered ``_nat_code_agent_workflow``
    function in workflow.py).
"""

from dotenv import load_dotenv

from nemo_code_agent.workflow import build_graph

load_dotenv()

# Compiled graph for NAT — no checkpointer needed since NAT manages sessions.
# LLMs are sourced from environment variables at compile time (see module docstring).
graph = build_graph(checkpointer=None)
