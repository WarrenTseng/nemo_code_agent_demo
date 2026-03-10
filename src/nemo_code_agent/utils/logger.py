"""Background debug logger — writes to .agent_logs/debug.log.

Keeps verbose inter-agent payloads, guardrail events, and tool I/O out of the
Rich terminal UI.  Call ``get_logger(__name__)`` from any module.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(os.environ.get("CODE_AGENT_LOG_DIR", ".agent_logs"))
_LOG_FILE = _LOG_DIR / "debug.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 3
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

_root_configured = False


def _configure_root() -> None:
    global _root_configured
    if _root_configured:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("nemo_code_agent")
    root.setLevel(logging.DEBUG)

    # Rotating file handler — debug level, all agent logs land here
    fh = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(fh)

    # Silence noisy third-party loggers that would pollute the file
    for noisy in ("httpx", "httpcore", "openai", "langchain", "langgraph"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Suppress nemoguardrails internal errors/warnings — non-fatal failures are
    # handled gracefully in guardrails.py; only show critical failures.
    logging.getLogger("nemoguardrails").setLevel(logging.CRITICAL)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under ``nemo_code_agent``.

    The first call also wires up the rotating file handler so all subsequent
    ``get_logger`` calls share the same handler without duplication.
    """
    _configure_root()
    # Ensure the name is always scoped under our package
    if not name.startswith("nemo_code_agent"):
        name = f"nemo_code_agent.{name}"
    return logging.getLogger(name)
