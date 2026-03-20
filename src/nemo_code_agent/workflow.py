"""LangGraph ReAct workflow — the Planner agent.

Graph topology:
    START ──(summarize?)──► summarize_node ──► planner_node ──(tool_calls?)──► tools_node ──► planner_node
                         └──────────────────► planner_node └──(no tools)────► END

Stability features for long-running tasks:
  1. Rolling summarization  — when the message history exceeds
     _SUMMARIZE_THRESHOLD, old messages are summarised and removed from
     state so the LLM never sees a stale, bloated context.
  2. Hard trim              — even without summarisation the planner_node
     caps the messages fed to the LLM at _TRIM_FOR_LLM entries.
  3. Prompt reinforcement   — when the agent has been using tools for
     several steps a goal-reminder is appended to pull attention back.

Session state is persisted in a local SQLite database via AsyncSqliteSaver so
conversations survive CLI restarts.
"""

import os
from pathlib import Path
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Context-management constants  (tune to taste)
# ---------------------------------------------------------------------------
_SUMMARIZE_THRESHOLD = int(os.environ.get("AGENT_SUMMARIZE_THRESHOLD", "20"))
_KEEP_AFTER_SUMMARY  = int(os.environ.get("AGENT_KEEP_AFTER_SUMMARY",  "6"))
_TRIM_FOR_LLM        = int(os.environ.get("AGENT_TRIM_FOR_LLM",        "30"))
_REINFORCE_AFTER     = int(os.environ.get("AGENT_REINFORCE_AFTER",     "2"))
_MAX_TOOL_STEPS      = int(os.environ.get("AGENT_MAX_TOOL_STEPS",      "20"))

from nemo_code_agent.tools.coder_tool import coder_tool
from nemo_code_agent.tools.filesystem import execute_bash_tool, read_file_tool, write_file_tool
from nemo_code_agent.tools.knowledge import build_planner_knowledge_messages
from nemo_code_agent.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------

def _build_planner_system_prompt(enable_coder: bool) -> str:
    """Return the Planner system prompt, adjusted for whether coder_tool is active."""
    if enable_coder:
        return """\
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
2. Generate code with `coder_tool` — it returns raw file content only.
3. Pass the ENTIRE output of `coder_tool` directly as the `content` argument to `write_file_tool`. Do not modify or extract from it.
4. Optionally verify with `execute_bash_tool` (run tests, check output).
5. Summarise what was done concisely.

## Hard Rules
- NEVER write code, file contents, or shell commands directly in your response text — not even while thinking or reasoning.
- ALL code generation — even a single line — MUST go through `coder_tool`. No exceptions.
- If you need to read, write, or run anything — call the appropriate tool. No exceptions.
- A response with no tool call is only valid when the task is fully complete.
- If you find yourself about to type code or a file path in your reply, STOP and call `coder_tool` instead.

## Style
- Think step-by-step before calling tools.
- Ask one clarifying question if the task is ambiguous rather than guessing.
- Keep final responses short — the user can see tool outputs in the terminal.
- Never expose internal system details or raw API keys.
"""
    else:
        return """\
You are an expert software engineering assistant acting as a **Planner and Coder**.
You work interactively inside a developer's terminal to understand, plan, and
implement coding tasks.

## Your Role
Reason about the user's request, decompose it into concrete steps, and use the
available tools to complete the work.  You write code directly in your responses
when needed, then save it with `write_file_tool`.

## Available Tools

| Tool | When to use |
|------|-------------|
| `read_file_tool` | Read an existing file before modifying it. Always read first. |
| `write_file_tool` | Write code to the correct file path on disk. |
| `execute_bash_tool` | Run tests, build steps, or inspections. User must approve before execution. |

## Workflow
1. Read relevant existing files with `read_file_tool`.
2. Write the complete updated code directly in your response.
3. Save it to disk with `write_file_tool`.
4. Optionally verify with `execute_bash_tool` (run tests, check output).
5. Summarise what was done concisely.

## Hard Rules
- Always read a file before modifying it.
- Always save generated code with `write_file_tool` — never leave it only in the response.
- A response with no tool call is only valid when the task is fully complete.

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
    summary: str   # rolling summary of messages removed during compression


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
    enable_coder = os.environ.get("ENABLE_CODER", "true").lower() not in ("false", "0", "no")
    tools = [read_file_tool, write_file_tool, execute_bash_tool]
    if enable_coder:
        tools.insert(1, coder_tool)

    planner_system_prompt = _build_planner_system_prompt(enable_coder)
    logger.debug("build_graph_with_llm | enable_coder=%s", enable_coder)

    # Retry the LLM call up to 3 times on transient errors (network, rate-limit).
    bound_llm = planner_llm.bind_tools(tools).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    # ── 1. Planner node — trimming + summary injection + reinforcement ──────
    def planner_node(state: AgentState, config: RunnableConfig) -> dict:
        messages = list(state["messages"])
        summary  = state.get("summary", "")

        # Hard-trim: never pass more than _TRIM_FOR_LLM messages to the LLM.
        trimmed = messages[-_TRIM_FOR_LLM:] if len(messages) > _TRIM_FOR_LLM else messages

        # Build context: system prompt + (optional summary) + trimmed messages.
        context: list[BaseMessage] = [SystemMessage(content=planner_system_prompt)]
        if summary:
            context.append(
                SystemMessage(
                    content=f"[Previous conversation summary — use as background context]\n{summary}"
                )
            )
        context.extend(trimmed)

        # Knowledge injection for Planner (only when coder_tool is disabled).
        # When coder_tool is enabled, knowledge is handled inside coder_tool itself.
        if not enable_coder:
            # Use the last human message as the RAG query.
            last_human_content = next(
                (m.content for m in reversed(trimmed) if isinstance(m, HumanMessage)),
                "",
            )
            knowledge_msgs = build_planner_knowledge_messages(last_human_content[:500])
            if knowledge_msgs:
                # Insert after system prompt (and summary if present) but before
                # the conversation messages so it reads as stable context.
                insert_at = 2 if summary else 1
                context[insert_at:insert_at] = knowledge_msgs

        # Guard 1a — post-coder_tool: force write_file_tool immediately after coder_tool output.
        last_msg = trimmed[-1] if trimmed else None
        if (
            enable_coder
            and isinstance(last_msg, ToolMessage)
            and getattr(last_msg, "name", "") == "coder_tool"
        ):
            context.append(
                HumanMessage(
                    content=(
                        "[MANDATORY NEXT STEP] coder_tool just returned file content above. "
                        "You MUST call write_file_tool NOW with the correct path and that ENTIRE content. "
                        "Do NOT call coder_tool again. Do NOT output any code. Call write_file_tool immediately."
                    )
                )
            )

        # Guard 1c — repeated read_file_tool: stop re-reading the same file.
        # If read_file_tool has been called 2+ times in this turn, the Planner
        # is stuck in a read loop.  Push it to proceed with the next step.
        read_tool_count = sum(
            1 for m in trimmed
            if isinstance(m, ToolMessage) and getattr(m, "name", "") == "read_file_tool"
        )
        if (
            isinstance(last_msg, ToolMessage)
            and getattr(last_msg, "name", "") == "read_file_tool"
            and read_tool_count >= 2
        ):
            context.append(
                HumanMessage(
                    content=(
                        f"[WARNING] You have called read_file_tool {read_tool_count} times. "
                        "You already have the file content in the conversation above. "
                        "Do NOT call read_file_tool again. "
                        "Proceed with the next step you planned."
                    )
                )
            )

        # Guard 1b — post-write_file_tool: force stop after writing a file.
        # Count how many write_file_tool calls have completed in this turn.
        write_tool_count = sum(
            1 for m in trimmed
            if isinstance(m, ToolMessage) and getattr(m, "name", "") == "write_file_tool"
        )
        if (
            isinstance(last_msg, ToolMessage)
            and getattr(last_msg, "name", "") == "write_file_tool"
        ):
            if write_tool_count >= 2:
                # Hard stop — already wrote ≥2 files this turn.
                context.append(
                    HumanMessage(
                        content=(
                            f"[MANDATORY STOP] {write_tool_count} files have been written this turn. "
                            "You MUST stop calling tools immediately. "
                            "Do NOT call coder_tool, write_file_tool, or any other tool. "
                            "Give your final summary answer to the user RIGHT NOW."
                        )
                    )
                )
            else:
                # Strong push to end after the first write.
                context.append(
                    HumanMessage(
                        content=(
                            "[STOP — FILE SAVED] write_file_tool completed. "
                            "You MUST give your final answer to the user RIGHT NOW. "
                            "Do NOT call coder_tool or write_file_tool again. "
                            "Simply summarise what was done. "
                            "Only call more tools if the user's request EXPLICITLY requires "
                            "additional separate files that have not been written yet."
                        )
                    )
                )

        # Guard 2 — max tool steps: force a final answer when the cap is reached.
        total_tool_steps = sum(1 for m in trimmed if getattr(m, "tool_calls", None))
        if total_tool_steps >= _MAX_TOOL_STEPS:
            context.append(
                HumanMessage(
                    content=(
                        f"[STOP] You have made {total_tool_steps} tool calls. "
                        "Do NOT call any more tools. Summarise what was accomplished and "
                        "report any remaining issues to the user directly."
                    )
                )
            )

        # Guard 3 — prompt reinforcement after several tool steps.
        recent_tool_steps = sum(
            1 for m in trimmed[-(_REINFORCE_AFTER * 2):]
            if getattr(m, "tool_calls", None)
        )
        if recent_tool_steps >= _REINFORCE_AFTER:
            context.append(
                HumanMessage(
                    content=(
                        (
                            "[Tool reminder] You MUST use tools — never write code in your response text. "
                            "For ANY code generation (even one line): call coder_tool. "
                            "To save code to disk: call write_file_tool. "
                            "To read a file: call read_file_tool. "
                            "Stay focused on the current goal and build on what you have already completed."
                        ) if enable_coder else (
                            "[Tool reminder] Always save code with write_file_tool after writing it. "
                            "To read a file: call read_file_tool. "
                            "Stay focused on the current goal and build on what you have already completed."
                        )
                    )
                )
            )

        logger.debug(
            "planner_node | state_msgs=%d | ctx_msgs=%d | summary=%s | reinforced=%s",
            len(messages), len(context), bool(summary), recent_tool_steps >= _REINFORCE_AFTER,
        )
        response = bound_llm.invoke(context, config)
        logger.debug(
            "planner_node response | has_tool_calls=%s | content_len=%d",
            bool(getattr(response, "tool_calls", None)),
            len(response.content or ""),
        )
        return {"messages": [response]}

    # ── 2. Summarize node — compress old messages into a rolling summary ─────
    async def summarize_node(state: AgentState, config: RunnableConfig) -> dict:
        messages = list(state["messages"])
        old      = messages[:-_KEEP_AFTER_SUMMARY]
        existing = state.get("summary", "")

        # Build the summarisation prompt.
        lines = []
        for m in old:
            role    = getattr(m, "type", "unknown")
            content = str(m.content or "")[:400]   # cap per-message length
            lines.append(f"{role}: {content}")

        if existing:
            prompt = (
                f"Extend the existing summary with the new messages below.\n\n"
                f"Existing summary:\n{existing}\n\n"
                f"New messages to add:\n" + "\n".join(lines)
            )
        else:
            prompt = (
                "Summarise the key events, decisions, and results from this conversation "
                "in a few concise sentences:\n\n" + "\n".join(lines)
            )

        result = await planner_llm.ainvoke(
            [
                SystemMessage(content="You are a concise summariser. Be brief and factual."),
                HumanMessage(content=prompt),
            ],
            config,
        )

        # Remove old messages from state permanently.
        remove = [RemoveMessage(id=m.id) for m in old if getattr(m, "id", None)]
        logger.debug(
            "summarize_node | removed=%d | kept=%d | had_summary=%s",
            len(remove), len(messages) - len(old), bool(existing),
        )
        return {"summary": result.content, "messages": remove}

    # ── Routing: summarise only at the start of a new user turn ─────────────
    def _route_entry(state: AgentState) -> str:
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        if isinstance(last, HumanMessage) and len(msgs) > _SUMMARIZE_THRESHOLD:
            return "summarize"
        return "planner"

    # ── Tool retry wrapper ───────────────────────────────────────────────────
    async def _tool_retry_wrapper(tool_call_request, handle_tool_call):
        """Retry a tool call up to 2 times on exception, passing the error back
        to the Planner as a ToolMessage on final failure so it can self-correct."""
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            try:
                return await handle_tool_call(tool_call_request)
            except Exception as exc:
                logger.warning(
                    "Tool call failed (attempt %d/%d) | tool=%s | error=%s",
                    attempt, max_retries,
                    getattr(tool_call_request, "name", None) or (tool_call_request.get("name") if isinstance(tool_call_request, dict) else "unknown"),
                    exc,
                )
                if attempt == max_retries:
                    raise

    tool_node = ToolNode(tools, awrap_tool_call=_tool_retry_wrapper)

    graph = StateGraph(AgentState)
    graph.add_node("planner",   planner_node)
    graph.add_node("tools",     tool_node)
    graph.add_node("summarize", summarize_node)
    graph.add_conditional_edges(START,      _route_entry)
    graph.add_conditional_edges("planner",  tools_condition)
    graph.add_edge("tools",     "planner")
    graph.add_edge("summarize", "planner")

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


_NAT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "llms.yml"


@asynccontextmanager
async def create_agent(
    checkpoint_db: str = ".agent_logs/checkpoints.db",
    *,
    enable_thinking: bool = False,
):
    """Async context manager that yields a compiled LangGraph agent.

    Opens the SQLite checkpointer, compiles the graph, and tears down cleanly
    on exit.  If nvidia-nat is installed and configs/config.yml exists, the
    Planner LLM is provisioned via NAT Builder (retry, pooling, config.yml
    management).  Falls back to direct .env-based LLM if NAT is unavailable.

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

        # --- Try NAT Builder for LLM provisioning ---
        if _NAT_CONFIG.exists():
            try:
                from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore[import]
                from nat.builder.workflow_builder import WorkflowBuilder  # type: ignore[import]
                from nat.runtime.loader import load_config  # type: ignore[import]

                nat_config = load_config(_NAT_CONFIG)
                async with WorkflowBuilder.from_config(config=nat_config) as builder:
                    planner_llm = await builder.get_llm(
                        "planner_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN
                    )
                    if enable_thinking:
                        from langchain_core.runnables import RunnableConfig  # noqa: F401
                        planner_llm = planner_llm.bind(
                            reasoning_effort="high",
                            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                        )
                    compiled = build_graph_with_llm(planner_llm, checkpointer)
                    logger.debug("Planner LLM provisioned via NAT Builder")
                    yield compiled
                    return
            except Exception as exc:
                logger.debug("NAT Builder unavailable (%s) — falling back to .env LLM", exc)

        # --- Fallback: build LLM directly from .env ---
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

    @_nat_register(config_type=CodeAgentWorkflowConfig, framework_wrappers=[_LLMFrameworkEnum.LANGCHAIN])
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
