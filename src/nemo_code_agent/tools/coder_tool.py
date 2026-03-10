"""Coder Tool — delegates code generation to a specialist Coding LLM.

The Coder is intentionally kept as a *tool* (not a top-level graph node) so that
the Planner drives all coordination.  The Coder LLM is a separate OpenAI-compatible
endpoint (e.g. local vLLM running a code model, or a NIM code endpoint) configured
via the CODER_* environment variables.
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from nemo_code_agent.utils.logger import get_logger

logger = get_logger(__name__)

_CODER_SYSTEM_PROMPT = """\
You are an expert software engineer specialising in writing clean, correct, and
well-structured code.

## Instructions
- Produce complete, runnable code for the requested task.
- If you are modifying an existing file, output the **full** updated file content.
- Always prefix each code block with the target file path as a comment, e.g.:
    # File: src/mypackage/utils.py
- After the code, add a brief **Changes** section explaining what was done and why.
- Do NOT add unnecessary prose — be concise and precise.
- Use type hints (Python), docstrings only for public APIs, and follow PEP-8.
"""


def _get_coder_llm() -> ChatOpenAI:
    """Instantiate the Coder LLM from environment config.

    Using a factory so the client is created fresh per invocation, which keeps
    connection pooling simple and avoids stale state between tool calls.
    """
    return ChatOpenAI(
        base_url=os.environ["CODER_URL"],
        model=os.environ["CODER_MODEL"],
        api_key=os.environ.get("CODER_API_KEY") or "none",
        temperature=float(os.environ.get("CODER_TEMPERATURE", "0.1")),
        max_tokens=int(os.environ.get("CODER_MAX_TOKENS", "16384")),
        # Streaming is not used here; the Planner streams its own tokens.
        streaming=False,
    )


@tool
def coder_tool(task: str) -> str:
    """Generate, modify, or debug code by delegating to a specialist Coding LLM.

    Use this tool for ALL code writing and modification tasks — do not write
    code yourself.  Provide a detailed, self-contained prompt that includes:
      - The programming language and relevant framework / library versions.
      - Exact file paths to create or modify.
      - Existing code snippets (paste them in full if short, summarise if long).
      - Clear acceptance criteria: what the finished code must do.
      - Any constraints (style guide, must not break existing tests, etc.).

    The tool returns the generated code with a brief explanation of changes.

    Args:
        task: Complete description of the coding task with all context needed.

    Returns:
        Generated code blocks (each prefixed with the target file path) followed
        by a short "Changes" summary.
    """
    logger.debug("coder_tool invoked | task length=%d chars", len(task))

    llm = _get_coder_llm()
    messages = [
        SystemMessage(content=_CODER_SYSTEM_PROMPT),
        HumanMessage(content=task),
    ]

    try:
        response = llm.invoke(messages)
        result = response.content
        logger.debug("coder_tool response | length=%d chars", len(result))
        return result
    except Exception as exc:
        logger.exception("coder_tool LLM call failed")
        return f"[coder_tool ERROR] {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# NAT plugin registration (requires: pip install "nemo-code-agent[nat]")
#
# When nvidia-nat is installed, this block registers coder_tool into the NAT
# plugin registry so it is usable via ``nat run --config configs/config.yml``.
# The NAT version uses Builder.get_llm() instead of the direct ChatOpenAI
# factory, letting NAT control connection pooling and retries.
# ---------------------------------------------------------------------------

try:
    from nat.builder.builder import Builder
    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.builder.function_info import FunctionInfo
    from nat.cli.register_workflow import register_function as _nat_register
    from nat.data_models.component_ref import LLMRef
    from nat.data_models.function import FunctionBaseConfig
    from pydantic import Field as _Field

    class CoderToolConfig(FunctionBaseConfig, name="coder_tool"):
        """NAT config for the Coder tool.  References ``coder_llm`` by default."""

        llm: LLMRef = _Field(
            default="coder_llm",
            description="NAT LLM reference for code generation (maps to coder_llm in config.yml).",
        )

    @_nat_register(config_type=CoderToolConfig)
    async def _nat_coder_tool(config: CoderToolConfig, builder: Builder):  # type: ignore[misc]
        async def _generate(task: str) -> str:
            """Generate, modify, or debug code using a specialist Coding LLM.

            Provide a complete, self-contained prompt with file paths, existing
            code snippets, and clear acceptance criteria.

            Args:
                task: Full description of the coding task with all context needed.

            Returns:
                Generated code blocks prefixed with target file paths, plus a
                brief Changes summary.
            """
            logger.debug("NAT coder_tool invoked | task_len=%d", len(task))
            llm = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

            response = await llm.ainvoke(
                [SystemMessage(content=_CODER_SYSTEM_PROMPT), HumanMessage(content=task)]
            )
            return response.content

        yield FunctionInfo.from_fn(_generate, description=_generate.__doc__)

    logger.debug("NAT coder_tool registered successfully")

except ImportError:
    pass  # nvidia-nat not installed — NAT registration skipped, CLI path unaffected
