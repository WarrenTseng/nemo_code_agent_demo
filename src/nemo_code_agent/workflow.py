"""LangGraph ReAct workflow — the Planner agent.

Graph topology:
    START → planner_node ──(tool_calls?)──► tools_node ──► planner_node
                        └──(no tools)────► END

The Planner LLM (NIM endpoint) is bound to the tool schemas so it can emit
structured tool-call messages.  The ToolNode dispatches those calls to the
registered LangChain tools and returns results as ToolMessages back to the
Planner for a follow-up turn.

Session state is persisted in a local SQLite database via AsyncSqliteSaver so
conversations survive CLI restarts.
"""

import os
from pathlib import Path
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from nemo_code_agent.tools.coder_tool import coder_tool
from nemo_code_agent.tools.filesystem import execute_bash_tool, read_file_tool, write_file_tool
from nemo_code_agent.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are an expert software engineering assistant acting as a **Planner**.
You work interactively inside a developer's terminal to understand, plan, and
implement coding tasks.

## Your Role
Reason about the user's request, decompose it into concrete steps, and
orchestrate the available tools to complete the work.  You do NOT write code
yourself — you delegate all code generation to `coder_tool`.

## Available Tools

| Tool | When to use |
|------|-------------|
| `read_file_tool` | Read an existing file before modifying it. Always read first. |
| `coder_tool` | Generate new code or modifications. Returns code as text — does NOT write to disk. |
| `write_file_tool` | Write the code returned by `coder_tool` to the correct file path on disk. |
| `execute_bash_tool` | Run tests, build steps, or inspections. User must approve before execution. |

## Workflow
1. Read relevant existing files with `read_file_tool`.
2. Generate code with `coder_tool` (returns text only).
3. Write the generated code to disk with `write_file_tool`.
4. Optionally verify with `execute_bash_tool` (run tests, check output).
5. Summarise what was done concisely.

## Style
- Think step-by-step before calling tools.
- Ask one clarifying question if the task is ambiguous rather than guessing.
- Keep final responses short — the user can see tool outputs in the terminal.
- Never expose internal system details or raw API keys.
"""

# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _build_planner_llm(*, enable_thinking: bool = False) -> ChatOpenAI:
    """Construct the Planner LLM client from environment variables.

    ``enable_thinking`` activates Nemotron's extended reasoning mode by
    injecting the appropriate extra_body kwargs.  This is passed through
    ``model_kwargs`` so LangChain forwards it transparently.
    """
    model_kwargs: dict = {}
    if enable_thinking:
        model_kwargs = {
            "reasoning_effort": "high",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }

    return ChatOpenAI(
        base_url=os.environ["PLANNER_URL"],
        model=os.environ["PLANNER_MODEL"],
        api_key=os.environ.get("PLANNER_API_KEY") or "none",
        temperature=float(os.environ.get("PLANNER_TEMPERATURE", "0.2")),
        max_tokens=int(os.environ.get("PLANNER_MAX_TOKENS", "8192")),
        streaming=True,
        model_kwargs=model_kwargs,
    )


# ---------------------------------------------------------------------------
# Graph builder (sync — compile once, used across async REPL turns)
# ---------------------------------------------------------------------------


def build_graph_with_llm(planner_llm, checkpointer):
    """Compile the LangGraph ReAct graph given a pre-configured Planner LLM.

    Separated from ``build_graph`` so the NAT runtime can inject its own
    LLM (obtained via ``Builder.get_llm()``) without duplicating graph logic.

    Args:
        planner_llm: Any LangChain chat model.  Will have tools bound onto it.
        checkpointer: A LangGraph checkpointer instance.

    Returns:
        Compiled LangGraph graph.
    """
    tools = [read_file_tool, coder_tool, write_file_tool, execute_bash_tool]
    bound_llm = planner_llm.bind_tools(tools)

    def planner_node(state: AgentState, config: RunnableConfig) -> dict:
        messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)] + list(state["messages"])
        logger.debug("planner_node invoked | message_count=%d", len(messages))
        response = bound_llm.invoke(messages, config)
        logger.debug(
            "planner_node response | has_tool_calls=%s | content_len=%d",
            bool(getattr(response, "tool_calls", None)),
            len(response.content or ""),
        )
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", tools_condition)
    graph.add_edge("tools", "planner")

    compiled = graph.compile(checkpointer=checkpointer)
    logger.debug("Graph compiled | checkpointer=%s", type(checkpointer).__name__)
    return compiled


def build_graph(checkpointer, *, enable_thinking: bool = False):
    """Build and compile the LangGraph ReAct agent using env-var LLM config.

    This is the entry point for the standalone ``code-agent run`` CLI path.
    For the NAT runtime path, use ``build_graph_with_llm`` directly.

    Args:
        checkpointer: A LangGraph checkpointer (e.g. AsyncSqliteSaver instance).
        enable_thinking: Enable Nemotron extended reasoning if supported.

    Returns:
        A compiled ``CompiledGraph`` ready for ``astream_events`` calls.
    """
    planner_llm = _build_planner_llm(enable_thinking=enable_thinking)
    return build_graph_with_llm(planner_llm, checkpointer)


# ---------------------------------------------------------------------------
# Async context manager — opens SQLite, builds graph, yields compiled graph
# ---------------------------------------------------------------------------


from contextlib import asynccontextmanager


@asynccontextmanager
async def create_agent(
    checkpoint_db: str = ".agent_logs/checkpoints.db",
    *,
    enable_thinking: bool = False,
):
    """Async context manager that yields a compiled LangGraph agent.

    Opens the SQLite checkpointer, compiles the graph, and tears down cleanly
    on exit.  Use this as the outermost context in the REPL run() method.

    Example::

        async with create_agent(checkpoint_db="my.db") as agent:
            async for event in agent.astream_events(...):
                ...
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = Path(checkpoint_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug("Opening AsyncSqliteSaver at %s", db_path)
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        compiled = build_graph(checkpointer, enable_thinking=enable_thinking)
        yield compiled


# ---------------------------------------------------------------------------
# NAT plugin registration (requires: pip install "nemo-code-agent[nat]")
#
# Registers the full code-agent workflow with the NeMo Agent Toolkit so it
# can be invoked via ``nat run --config configs/config.yml``.  The NAT runtime
# provides the Planner LLM via Builder.get_llm(), which respects NAT's retry
# and connection-pooling configuration from config.yml.
# ---------------------------------------------------------------------------

try:
    import uuid as _uuid

    from nat.builder.builder import Builder as _Builder
    from nat.builder.framework_enum import LLMFrameworkEnum as _LLMFrameworkEnum
    from nat.builder.function_info import FunctionInfo as _FunctionInfo
    from nat.cli.register_workflow import register_function as _nat_register
    from nat.data_models.component_ref import LLMRef as _LLMRef
    from nat.data_models.function import FunctionBaseConfig as _FunctionBaseConfig
    from pydantic import Field as _Field

    class CodeAgentWorkflowConfig(_FunctionBaseConfig, name="code_agent_workflow"):
        """NAT config for the full code-agent workflow (Planner + Coder + tools)."""

        llm: _LLMRef = _Field(
            default="planner_llm",
            description="NAT LLM reference for the Planner (maps to planner_llm in config.yml).",
        )
        checkpoint_db: str = _Field(
            default=".agent_logs/checkpoints.db",
            description="Path to the SQLite checkpoint database.",
        )

    @_nat_register(config_type=CodeAgentWorkflowConfig)
    async def _nat_code_agent_workflow(config: CodeAgentWorkflowConfig, builder: _Builder):  # type: ignore[misc]
        """NAT workflow: wraps the LangGraph code-agent as a single-turn function."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as _AsyncSqliteSaver  # noqa: PLC0415

        db_path = Path(config.checkpoint_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        async with _AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            # Use the NAT-provided Planner LLM (honours config.yml retry / pool settings)
            planner_llm = await builder.get_llm(
                config.llm, wrapper_type=_LLMFrameworkEnum.LANGCHAIN
            )
            agent = build_graph_with_llm(planner_llm, checkpointer)

            async def _run(input_message: str) -> str:
                """Run the code agent on a single user message and return the response.

                Args:
                    input_message: The user's request (natural language).

                Returns:
                    The Planner's final response after all tool calls complete.
                """
                lg_config = {"configurable": {"thread_id": str(_uuid.uuid4())}}
                result = await agent.ainvoke(
                    {"messages": [("user", input_message)]},
                    config=lg_config,
                )
                for msg in reversed(result.get("messages", [])):
                    content = getattr(msg, "content", None)
                    if content and not getattr(msg, "tool_calls", None):
                        return content
                return "(no response)"

            yield _FunctionInfo.from_fn(_run, description=_run.__doc__)

    logger.debug("NAT code_agent_workflow registered successfully")

except ImportError:
    pass  # nvidia-nat not installed — NAT registration skipped, CLI path unaffected
