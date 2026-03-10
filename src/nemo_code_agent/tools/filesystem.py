"""Filesystem tools — read_file_tool, write_file_tool, and execute_bash_tool.

execute_bash_tool is intentionally *async* so it can yield control to the event
loop while waiting for the user confirmation prompt without blocking other async
tasks.  read_file_tool and write_file_tool are sync (fast I/O).
"""

import asyncio
import subprocess
import threading
from pathlib import Path

from langchain_core.tools import tool
from rich.console import Console as _Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from nemo_code_agent.utils.logger import get_logger

logger = get_logger(__name__)

_console = _Console()

# Max characters returned by read_file_tool before truncation
_MAX_READ_CHARS = 50_000

# Timeout in seconds for shell commands
_BASH_TIMEOUT = 60

# Max lines shown in the live log panel at once
_LOG_VISIBLE_LINES = 20

# ---------------------------------------------------------------------------
# Per-turn state shared with the REPL for interrupt coordination
# ---------------------------------------------------------------------------

# Set by execute_bash_tool while waiting for user confirmation.
# The ESC watcher in the REPL pauses (and restores terminal mode) during this
# window to avoid conflicting with the blocking input() call.
confirmation_active = threading.Event()

# Set by execute_bash_tool when the user declines a command.
# The ESC watcher in the REPL detects this and cancels the current agent turn.
_state: dict = {"decline_detected": False}


def reset_turn_state() -> None:
    """Reset per-turn shared state. Called by the REPL at the start of each turn."""
    _state["decline_detected"] = False
    confirmation_active.clear()


def is_declined() -> bool:
    """Return True if the user declined a bash command this turn."""
    return _state["decline_detected"]


# ---------------------------------------------------------------------------
# read_file_tool
# ---------------------------------------------------------------------------


@tool
def read_file_tool(path: str) -> str:
    """Read the contents of a file on the local filesystem.

    Use this before modifying any existing file so you always have the current
    source available to pass to coder_tool.

    Args:
        path: Absolute or relative path to the file to read.

    Returns:
        The full file contents as a string, or an error message if the file
        cannot be read.  Files larger than 50,000 characters are truncated with
        a notice.
    """
    logger.debug("read_file_tool | path=%s", path)

    p = Path(path).expanduser()
    if not p.exists():
        return f"[read_file_tool ERROR] File not found: {path}"
    if not p.is_file():
        return f"[read_file_tool ERROR] Path is not a regular file: {path}"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.exception("read_file_tool failed to read %s", path)
        return f"[read_file_tool ERROR] Could not read file: {exc}"

    if len(content) > _MAX_READ_CHARS:
        content = content[:_MAX_READ_CHARS]
        content += f"\n\n... [TRUNCATED — file exceeds {_MAX_READ_CHARS:,} chars] ..."
        logger.debug("read_file_tool truncated output for %s", path)

    return content


# ---------------------------------------------------------------------------
# write_file_tool
# ---------------------------------------------------------------------------


@tool
def write_file_tool(path: str, content: str) -> str:
    """Write content to a file on the local filesystem.

    Use this after coder_tool has returned generated code — pass the exact
    file path and the full file content (not a diff or patch).  Parent
    directories are created automatically.

    Args:
        path: Absolute or relative path of the file to write or overwrite.
        content: The complete file content to write.

    Returns:
        Confirmation with the resolved path and byte count, or an error message.
    """
    logger.debug("write_file_tool | path=%s | content_len=%d", path, len(content))

    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        byte_count = len(content.encode("utf-8"))
        p.write_text(content, encoding="utf-8")
        logger.debug("write_file_tool success | path=%s | bytes=%d", p, byte_count)
        return f"Written: {p.resolve()} ({byte_count:,} bytes)"
    except OSError as exc:
        logger.exception("write_file_tool failed | path=%s", path)
        return f"[write_file_tool ERROR] {exc}"


# ---------------------------------------------------------------------------
# execute_bash_tool
# ---------------------------------------------------------------------------


@tool
async def execute_bash_tool(command: str) -> str:
    """Execute a bash command and stream its output live in the terminal.

    IMPORTANT: The user will be shown the command and asked to approve it
    before it runs.  If the user declines, the command is NOT executed and
    the current agent turn is cancelled.

    Use for:
      - Running tests:  pytest, cargo test, npm test …
      - Building:       make, pip install -e ., npm run build …
      - Inspection:     ls, cat, python script.py …

    Avoid destructive one-liners (rm -rf, DROP TABLE, git reset --hard) unless
    the user has explicitly asked for them — they will still see the confirmation
    prompt, but prefer safer alternatives when possible.

    Args:
        command: The shell command to run (executed via bash -c).

    Returns:
        Combined stdout + stderr from the command, or a cancellation notice.
    """
    logger.debug("execute_bash_tool | command=%s", command)

    # --- Interactive confirmation -------------------------------------------
    _console.print()
    _console.print("[bold yellow]\\[Agent wants to run a command][/bold yellow]")
    _console.print(f"  [bold]$ {command}[/bold]")

    # Signal the ESC watcher to pause and restore normal terminal mode so that
    # the blocking input() call works correctly.
    confirmation_active.set()
    try:
        answer = await asyncio.to_thread(
            input,
            "\033[33mApprove? [Y/n]: \033[0m",
        )
    finally:
        confirmation_active.clear()

    if answer.strip().lower() in ("n", "no"):
        logger.info("execute_bash_tool declined by user | command=%s", command)
        _state["decline_detected"] = True
        return "Command cancelled by user."

    # --- Stream output with Rich Live panel ---------------------------------
    output_lines: list[str] = []

    def _make_panel() -> Panel:
        visible = output_lines[-_LOG_VISIBLE_LINES:]
        body = Text("\n".join(visible), no_wrap=False, overflow="fold")
        subtitle = (
            f"[dim]{len(output_lines)} lines total — showing last {_LOG_VISIBLE_LINES}[/dim]"
            if len(output_lines) > _LOG_VISIBLE_LINES
            else f"[dim]{len(output_lines)} lines[/dim]"
        )
        return Panel(
            body,
            title=f"[bold yellow]$ {command[:70]}[/bold yellow]",
            subtitle=subtitle,
            border_style="yellow",
        )

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        with Live(_make_panel(), console=_console, refresh_per_second=10) as live:
            assert proc.stdout is not None
            while True:
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=_BASH_TIMEOUT)
                except asyncio.TimeoutError:
                    proc.kill()
                    output_lines.append(f"[timed out after {_BASH_TIMEOUT}s]")
                    live.update(_make_panel())
                    break
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(decoded)
                live.update(_make_panel())

        await proc.wait()

        stdout = "\n".join(output_lines)
        result = stdout.strip() if stdout.strip() else "(command produced no output)"
        logger.debug(
            "execute_bash_tool completed | returncode=%d | lines=%d",
            proc.returncode,
            len(output_lines),
        )
        if proc.returncode != 0:
            result = f"[exit code {proc.returncode}]\n{result}"
        return result

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("execute_bash_tool unexpected error")
        return f"[execute_bash_tool ERROR] {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# NAT plugin registration (requires: pip install "nemo-code-agent[nat]")
# ---------------------------------------------------------------------------

try:
    from nat.builder.builder import Builder
    from nat.builder.function_info import FunctionInfo
    from nat.cli.register_workflow import register_function as _nat_register
    from nat.data_models.function import FunctionBaseConfig
    from pydantic import Field as _Field

    class ReadFileToolConfig(FunctionBaseConfig, name="read_file_tool"):
        """NAT config for the read_file tool."""

        max_chars: int = _Field(
            default=_MAX_READ_CHARS,
            description="Maximum characters to return before truncating.",
        )

    @_nat_register(config_type=ReadFileToolConfig)
    async def _nat_read_file_tool(config: ReadFileToolConfig, builder: Builder):  # type: ignore[misc]
        async def _read(path: str) -> str:
            """Read the contents of a file on the local filesystem.

            Use this before modifying any existing file so you always have the
            current source to pass to coder_tool.

            Args:
                path: Absolute or relative path to the file.

            Returns:
                File contents (truncated at max_chars if needed) or an error message.
            """
            logger.debug("NAT read_file_tool | path=%s", path)
            p = Path(path).expanduser()
            if not p.exists():
                return f"[read_file_tool ERROR] File not found: {path}"
            if not p.is_file():
                return f"[read_file_tool ERROR] Not a regular file: {path}"
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return f"[read_file_tool ERROR] {exc}"
            if len(content) > config.max_chars:
                content = content[: config.max_chars]
                content += f"\n\n... [TRUNCATED at {config.max_chars:,} chars] ..."
            return content

        yield FunctionInfo.from_fn(_read, description=_read.__doc__)

    class WriteFileToolConfig(FunctionBaseConfig, name="write_file_tool"):
        """NAT config for the write_file tool."""

    @_nat_register(config_type=WriteFileToolConfig)
    async def _nat_write_file_tool(config: WriteFileToolConfig, builder: Builder):  # type: ignore[misc]
        async def _write(path: str, content: str) -> str:
            """Write content to a file on the local filesystem.

            Creates parent directories as needed.  Use after coder_tool returns
            the generated code.

            Args:
                path: Absolute or relative path of the file to write.
                content: The complete file content to write.

            Returns:
                Confirmation with resolved path and byte count, or an error.
            """
            logger.debug("NAT write_file_tool | path=%s | content_len=%d", path, len(content))
            p = Path(path).expanduser()
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                byte_count = len(content.encode("utf-8"))
                p.write_text(content, encoding="utf-8")
                return f"Written: {p.resolve()} ({byte_count:,} bytes)"
            except OSError as exc:
                return f"[write_file_tool ERROR] {exc}"

        yield FunctionInfo.from_fn(_write, description=_write.__doc__)

    class ExecuteBashToolConfig(FunctionBaseConfig, name="execute_bash_tool"):
        """NAT config for the execute_bash tool."""

        require_confirmation: bool = _Field(
            default=True,
            description="Prompt the user for approval before running each command.",
        )
        timeout: int = _Field(
            default=_BASH_TIMEOUT,
            description="Shell command timeout in seconds.",
        )

    @_nat_register(config_type=ExecuteBashToolConfig)
    async def _nat_execute_bash_tool(config: ExecuteBashToolConfig, builder: Builder):  # type: ignore[misc]
        async def _execute(command: str) -> str:
            """Execute a bash command and return its combined stdout + stderr.

            The user will be prompted for confirmation before the command runs
            (when require_confirmation is true).

            Args:
                command: Shell command to run via bash -c.

            Returns:
                Combined stdout + stderr, or a cancellation/error notice.
            """
            logger.debug("NAT execute_bash_tool | command=%s", command)
            if config.require_confirmation:
                print()
                print(f"\033[33m[Agent wants to run a command]\033[0m")
                print(f"  \033[1m$ {command}\033[0m")
                answer = await asyncio.to_thread(input, "\033[33mApprove? [y/N]: \033[0m")
                if answer.strip().lower() not in ("y", "yes"):
                    return "Command cancelled by user."
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )
                output = ((proc.stdout or "") + (proc.stderr or "")).strip()
                result = output if output else "(no output)"
                if proc.returncode != 0:
                    result = f"[exit code {proc.returncode}]\n{result}"
                return result
            except subprocess.TimeoutExpired:
                return f"[execute_bash_tool ERROR] Timed out after {config.timeout}s."
            except Exception as exc:
                return f"[execute_bash_tool ERROR] {type(exc).__name__}: {exc}"

        yield FunctionInfo.from_fn(_execute, description=_execute.__doc__)

    logger.debug("NAT filesystem tools registered successfully")

except ImportError:
    pass  # nvidia-nat not installed — NAT registration skipped
