"""Output guardrails for the code agent.

Two-layer defence:

1. **Pattern check** (always active, zero dependencies):
   Fast regex scan for obviously destructive shell commands and SQL statements.
   Catches the canonical examples defined in ``guardrails/guardrails.co``.

2. **NeMo Guardrails** (active when ``nemoguardrails`` is installed):
   Loads ``guardrails/config.yml`` and runs the Colang input and output rail
   flows against user messages and Planner responses respectively.
   Registers the ``check_if_impolite`` action backed by a direct LLM call.
   If NeMo Guardrails modifies or rejects a message the blocked flag is set.

Both output layers run after the LangGraph stream is fully buffered, so the
user still sees tokens arriving in real time — the guardrails gate only the
final Markdown render.  If either layer blocks, the raw streamed text is
replaced with a safe notice and a warning panel is shown instead.

The input layer runs before the user message is sent to the Planner.

To extend the Colang rails, edit ``guardrails/guardrails.co``.
To add NeMo Guardrails:  ``pip install "nemo-code-agent[guardrails]"``
"""

import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple

from nemo_code_agent.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pattern-based check (Layer 1)
# ---------------------------------------------------------------------------

_RAW_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-rf\s+/", "rm -rf /"),
    (r"\brm\s+-rf\s+~", "rm -rf ~"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"\bdd\s+if=\S+\s+of=/dev/[sh]d", "dd to block device"),
    (r"\bmkfs\.\w+\s+/dev/", "mkfs to device"),
    (r"\bformat\s+[a-zA-Z]:\s*/[qy]", "Windows format /q"),
    (r"\bDROP\s+TABLE\b", "SQL DROP TABLE"),
    (r"\bDROP\s+DATABASE\b", "SQL DROP DATABASE"),
    (r"\bDELETE\s+FROM\b\s+\w+\s*;?\s*$", "unconstrained SQL DELETE"),
    (r"\btruncate\s+table\b", "SQL TRUNCATE TABLE"),
    (r">\s*/dev/sd[a-z]\b", "redirect to block device"),
]

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(raw, re.IGNORECASE | re.MULTILINE), label)
    for raw, label in _RAW_PATTERNS
]

_BLOCK_NOTICE = (
    "[Safety] This response was withheld because it contained a pattern "
    "that matches a potentially destructive operation. "
    "If this was intentional, please confirm with a more explicit request."
)

_INPUT_BLOCK_NOTICE = (
    "I'm here to help with coding tasks. "
    "Could you please rephrase your request in a more constructive way?"
)


def _pattern_check(response: str) -> Tuple[str, bool]:
    """Return (response, blocked). Blocked responses are replaced with a notice."""
    for pattern, label in _COMPILED:
        if pattern.search(response):
            logger.warning("Guardrails (pattern) blocked response | pattern=%s", label)
            return _BLOCK_NOTICE, True
    return response, False


# ---------------------------------------------------------------------------
# check_if_impolite — LLM-backed action (used by NeMo Guardrails Colang)
# ---------------------------------------------------------------------------

async def _llm_is_impolite(text: str) -> bool:
    """Call the Planner LLM to judge whether *text* is impolite or unprofessional.

    Returns True if the LLM considers the text impolite, False otherwise.
    Falls back to False (non-blocking) on any error so guardrails never
    crash the agent.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore[import]

        client = AsyncOpenAI(
            base_url=os.environ["PLANNER_URL"],
            api_key=os.environ.get("PLANNER_API_KEY") or "none",
        )
        resp = await client.chat.completions.create(
            model=os.environ["PLANNER_MODEL"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a content moderator. Answer only 'yes' or 'no'.",
                },
                {
                    "role": "user",
                    "content": (
                        "Is the following text impolite, rude, aggressive, or contains "
                        "profanity? Answer only yes or no.\n\nText: " + text
                    ),
                },
            ],
            max_tokens=5,
            temperature=0,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        return answer.startswith("yes")
    except Exception as exc:
        logger.warning("check_if_impolite LLM call failed (non-fatal): %s", exc)
        return False


async def check_if_impolite(text: str = "", **_kwargs: Any) -> bool:
    """NeMo Guardrails action — called via ``execute check_if_impolite(text=...)``."""
    return await _llm_is_impolite(text)


# ---------------------------------------------------------------------------
# NeMo Guardrails check (Layer 2 — optional)
# ---------------------------------------------------------------------------

_GUARDRAILS_CONFIG_DIR = Path(__file__).parent.parent.parent / "guardrails"

# Cached LLMRails instance — created once, reused across calls.
_rails_cache: Optional[Any] = None


async def _get_rails() -> Optional[Any]:
    """Return a cached ``LLMRails`` instance, or None if unavailable."""
    global _rails_cache
    if _rails_cache is not None:
        return _rails_cache

    # NeMo Guardrails is opt-in — set GUARDRAILS_ENABLED=true to activate.
    if os.environ.get("GUARDRAILS_ENABLED", "false").lower() != "true":
        logger.debug("NeMo Guardrails disabled (GUARDRAILS_ENABLED != true)")
        return None

    try:
        from nemoguardrails import LLMRails, RailsConfig  # type: ignore[import]
    except Exception as e:
        logger.debug("NeMo Guardrails unavailable (%s: %s) — skipping", type(e).__name__, e)
        return None

    if not _GUARDRAILS_CONFIG_DIR.exists():
        logger.debug("NeMo Guardrails config dir not found at %s — skipping", _GUARDRAILS_CONFIG_DIR)
        return None

    try:
        # Expand environment variables in config.yml before loading.
        # RailsConfig.from_path() reads the file as-is and does NOT substitute
        # ${VAR} placeholders — we must do it ourselves.
        yaml_content = os.path.expandvars(
            (_GUARDRAILS_CONFIG_DIR / "config.yml").read_text()
        )
        colang_content = (_GUARDRAILS_CONFIG_DIR / "guardrails.co").read_text()
        rails_config = RailsConfig.from_content(
            yaml_content=yaml_content,
            colang_content=colang_content,
        )
        rails = LLMRails(rails_config)
        rails.register_action(check_if_impolite, name="check_if_impolite")
        _rails_cache = rails
        logger.debug("NeMo Guardrails initialised from %s", _GUARDRAILS_CONFIG_DIR)
        return rails
    except Exception as exc:
        logger.debug("NeMo Guardrails init failed (non-fatal): %s", exc)
        return None


async def _nemo_check(response: str) -> Tuple[str, bool]:
    """Run NeMo Guardrails output rails on the buffered Planner response."""
    rails = await _get_rails()
    if rails is None:
        return response, False

    try:
        checked = await rails.generate_async(
            messages=[
                {"role": "user", "content": "[output safety check]"},
                {"role": "assistant", "content": response},
            ]
        )

        if isinstance(checked, dict):
            checked = checked.get("content", response)

        if checked != response:
            logger.warning("NeMo Guardrails modified the output response")
            return str(checked), True

        return response, False

    except Exception as exc:
        logger.debug("NeMo Guardrails output check failed (non-fatal): %s", exc)
        return response, False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


async def apply_input_guardrails(user_input: str) -> Tuple[str, bool]:
    """Check user input before it is sent to the Planner.

    Runs the LLM-based politeness check when NeMo Guardrails is installed.
    Falls back silently if not installed.

    Returns:
        ``(user_input, False)`` if the message is clean.
        ``(notice, True)`` if blocked — the REPL shows the notice instead.
    """
    rails = await _get_rails()
    if rails is None:
        return user_input, False

    is_impolite = await _llm_is_impolite(user_input)
    if is_impolite:
        logger.warning("Input guardrails blocked impolite user message")
        return _INPUT_BLOCK_NOTICE, True

    return user_input, False


async def apply_guardrails(response: str) -> Tuple[str, bool]:
    """Run both guardrails layers on the Planner's buffered response.

    Called by the REPL after the stream completes, before the Markdown render.

    Returns:
        ``(final_response, was_blocked)`` — if blocked, ``final_response`` is a
        safe replacement notice and ``was_blocked`` is ``True``.
    """
    response, blocked = _pattern_check(response)
    if blocked:
        return response, True

    response, blocked = await _nemo_check(response)
    return response, blocked
