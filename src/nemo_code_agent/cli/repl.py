"""Interactive REPL — prompt_toolkit input + Rich output.

Layout:
  - prompt_toolkit ``PromptSession`` handles multi-line aware input with history.
  - ``rich.Console`` renders all agent output (streaming tokens, tool banners,
    reasoning traces, errors).
  - ``astream_events(version="v2")`` drives real-time streaming from LangGraph.

Streaming rendering strategy:
  - Planner reasoning tokens (``reasoning_content``) → dim italic grey.
  - Planner response tokens (``content``) → bright white, accumulate in buffer.
  - Tool start → yellow rule banner with tool name.
  - Tool end → dim closing rule; full I/O goes to debug.log only.
  - Final response rendered as Markdown after streaming completes.

Interruption:
  - Press ESC at any time during streaming to cancel the current agent turn.
  - Declining a bash command (typing "n") also cancels the turn immediately.
  - Both return cleanly to the chat prompt.
"""

import asyncio
import os
import sys
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from nemo_code_agent.guardrails import apply_guardrails, apply_input_guardrails
from nemo_code_agent.tools import filesystem as _fs
from nemo_code_agent.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_PROMPT_STYLE = Style.from_dict(
    {
        "prompt.user": "ansicyan bold",
        "bottom-toolbar": "bg:#1a1a1a #888888",
    }
)

_BOTTOM_TOOLBAR = HTML(" <b>ESC</b> interrupt  |  <b>Tab</b> autocomplete  |  <b>&#8593;&#8595;</b> history  |  <b>Ctrl-D</b> quit")

_PROMPT_TEXT = HTML("<ansicyan><b>You ❯ </b></ansicyan>")

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# History-based Tab completer
# ---------------------------------------------------------------------------


class _HistoryCompleter(Completer):
    """Completes the current input from previous history entries on Tab press."""

    def __init__(self, history: FileHistory) -> None:
        self._history = history

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.strip():
            return
        seen: set[str] = set()
        for entry in self._history.load_history_strings():
            if entry.startswith(text) and entry != text and entry not in seen:
                seen.add(entry)
                yield Completion(
                    entry[len(text):],
                    start_position=0,
                    display=entry,
                )


# ---------------------------------------------------------------------------
# ESC watcher
# ---------------------------------------------------------------------------


async def _watch_for_esc(cancel_event: asyncio.Event) -> None:
    """Poll stdin for an ESC keypress while the agent is streaming.

    Uses cbreak mode + non-blocking select so we never block the event loop.
    Automatically pauses (and restores normal terminal mode) while
    ``execute_bash_tool`` is waiting for the user's confirmation input, then
    re-enters cbreak mode afterwards.

    Sets *cancel_event* on ESC press or when a bash command is declined.
    Falls back gracefully on platforms without termios (e.g. Windows).
    """
    try:
        import select
        import termios
        import tty
    except ImportError:
        # termios not available (Windows) — ESC interrupt not supported
        while not cancel_event.is_set():
            await asyncio.sleep(0.1)
            if _fs.is_declined():
                cancel_event.set()
        return

    fd = sys.stdin.fileno()

    # If stdin is not a real TTY (e.g. piped input), skip cbreak mode.
    if not sys.stdin.isatty():
        while not cancel_event.is_set():
            await asyncio.sleep(0.1)
            if _fs.is_declined():
                cancel_event.set()
        return

    old_settings = termios.tcgetattr(fd)

    def _enter_cbreak() -> None:
        tty.setcbreak(fd)

    def _restore() -> None:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    try:
        _enter_cbreak()

        while not cancel_event.is_set():
            # ------------------------------------------------------------------
            # Pause ESC detection while execute_bash_tool awaits confirmation.
            # Restore normal terminal mode so input() works correctly.
            # ------------------------------------------------------------------
            if _fs.confirmation_active.is_set():
                _restore()
                while _fs.confirmation_active.is_set():
                    await asyncio.sleep(0.05)
                    if _fs.is_declined():
                        cancel_event.set()
                        return
                # Confirmation done — check decline one more time
                if _fs.is_declined():
                    cancel_event.set()
                    return
                # Re-enter cbreak for continued ESC monitoring
                await asyncio.sleep(0.05)  # let input() thread fully finish
                _enter_cbreak()
                continue

            await asyncio.sleep(0.05)

            # Non-blocking check: any byte waiting in stdin?
            r, _, _ = select.select([fd], [], [], 0)
            if r:
                char = os.read(fd, 1)
                if char == b"\x1b":  # ESC
                    cancel_event.set()
                    return

            # Also check for bash decline even if no keypress
            if _fs.is_declined():
                cancel_event.set()
                return

    except asyncio.CancelledError:
        pass
    finally:
        _restore()


# ---------------------------------------------------------------------------
# REPL class
# ---------------------------------------------------------------------------


class CodeAgentREPL:
    """Manages the interactive coding agent session.

    Args:
        agent: A compiled LangGraph graph (from ``create_agent`` context manager).
        session_id: Unique thread ID for LangGraph checkpointing.  Passing the
            same ID on restart resumes the previous conversation.
        history_file: Path to prompt_toolkit history file for up-arrow recall.
    """

    def __init__(
        self,
        agent,
        session_id: str,
        history_file: str = ".agent_logs/.repl_history",
    ) -> None:
        self.agent = agent
        self.session_id = session_id
        self._langgraph_config = {"configurable": {"thread_id": session_id}}

        _history = FileHistory(history_file)
        self._prompt_session: PromptSession = PromptSession(
            history=_history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=ThreadedCompleter(_HistoryCompleter(_history)),
            complete_while_typing=False,  # only activate on Tab, not on every keystroke
            style=_PROMPT_STYLE,
        )

    # ------------------------------------------------------------------
    # Streaming renderer
    # ------------------------------------------------------------------

    async def _stream_response(self, user_input: str) -> None:
        """Stream graph events and render them to the terminal in real time."""

        response_buffer: list[str] = []
        in_thinking: bool = False          # True while emitting reasoning tokens
        emitted_agent_prefix: bool = False  # Avoid double-printing "Agent ❯"
        current_tool: Optional[str] = None

        console.print()  # blank line before agent output

        try:
            async for event in self.agent.astream_events(
                {"messages": [("user", user_input)]},
                config=self._langgraph_config,
                version="v2",
            ):
                kind: str = event["event"]
                node: str = event.get("metadata", {}).get("langgraph_node", "")

                # ----------------------------------------------------------
                # Streaming tokens from the Planner LLM
                # ----------------------------------------------------------
                if kind == "on_chat_model_stream" and node == "planner":
                    chunk = event["data"]["chunk"]

                    # Nemotron / DeepSeek-R1-style thinking tokens
                    thinking: str = chunk.additional_kwargs.get("reasoning_content", "")
                    if thinking:
                        if not in_thinking:
                            console.print(
                                Text("⟨thinking⟩", style="dim italic"),
                                end=" ",
                            )
                            in_thinking = True
                        console.print(thinking, end="", style="dim italic grey50")

                    # Regular response tokens
                    content: str = chunk.content or ""
                    if content:
                        if in_thinking:
                            # Transition out of thinking block
                            console.print(
                                Text("\n⟨/thinking⟩\n", style="dim italic"),
                            )
                            in_thinking = False
                        if not emitted_agent_prefix:
                            console.print(
                                Text("Agent ❯ ", style="bold green"),
                                end="",
                            )
                            emitted_agent_prefix = True
                        console.print(content, end="", style="bright_white")
                        response_buffer.append(content)

                # ----------------------------------------------------------
                # Tool lifecycle events
                # ----------------------------------------------------------
                elif kind == "on_tool_start":
                    if in_thinking:
                        console.print(Text("\n⟨/thinking⟩\n", style="dim italic"))
                        in_thinking = False

                    current_tool = event["name"]
                    tool_input = event["data"].get("input", {})
                    console.print()
                    console.print(
                        Rule(
                            title=f"[bold yellow]▶ {current_tool}[/bold yellow]",
                            style="yellow",
                        )
                    )
                    # Show a truncated preview of the tool input
                    preview = str(tool_input)
                    if len(preview) > 200:
                        preview = preview[:200] + " …"
                    console.print(Text(preview, style="dim yellow"))
                    logger.debug(
                        "tool_start | tool=%s | input=%s",
                        current_tool,
                        str(tool_input)[:1000],
                    )

                elif kind == "on_tool_end":
                    output = event["data"].get("output", "")
                    console.print(
                        Rule(style="yellow dim"),
                    )
                    logger.debug(
                        "tool_end | tool=%s | output=%s",
                        current_tool,
                        str(output)[:1000],
                    )
                    current_tool = None
                    emitted_agent_prefix = False  # Reset so next Planner turn gets prefix

        except asyncio.CancelledError:
            # Clean cancellation — ESC pressed or command declined
            raise
        except Exception as exc:
            logger.exception("Error during agent stream")
            console.print(f"\n[bold red]Stream error:[/bold red] {exc}")
            return

        # ------------------------------------------------------------------
        # Post-stream: guardrails → optional Markdown render
        # ------------------------------------------------------------------
        full_response = "".join(response_buffer)
        console.print()  # newline after streamed tokens

        if full_response.strip():
            # Run both guardrail layers on the buffered response
            safe_response, was_blocked = await apply_guardrails(full_response)

            if was_blocked:
                # Replace the streamed output with a warning panel
                console.print(
                    Panel(
                        Text(safe_response, style="bold red"),
                        title="[bold red]⚠ Safety Check[/bold red]",
                        border_style="red",
                        padding=(0, 1),
                    )
                )
                logger.warning("Response blocked by guardrails for session=%s", self.session_id)
            else:
                # Re-render as Markdown if the response has structural markers
                if any(m in safe_response for m in ("```", "##", "**", "- ")):
                    console.print(
                        Panel(
                            Markdown(safe_response),
                            border_style="bright_black",
                            padding=(0, 1),
                        )
                    )
        else:
            # No text content — the agent may have only made tool calls
            if not in_thinking:
                console.print(
                    Text("[Agent completed without text response]", style="dim italic")
                )

        console.print()  # trailing blank line

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the interactive REPL loop."""
        console.print(
            Panel.fit(
                "[bold cyan]NeMo Code Agent[/bold cyan]\n"
                f"[dim]Session ID : {self.session_id}[/dim]\n"
                "[dim]Type [bold]exit[/bold] or press Ctrl-D to quit. "
                "Press [bold]ESC[/bold] to interrupt the agent.[/dim]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        console.print()

        while True:
            # ---- Read user input ----------------------------------------
            try:
                user_input: str = await self._prompt_session.prompt_async(
                    _PROMPT_TEXT,
                    bottom_toolbar=_BOTTOM_TOOLBAR,
                )
            except KeyboardInterrupt:
                console.print("\n[dim](Interrupted — type 'exit' to quit)[/dim]")
                continue
            except EOFError:
                console.print("\n[dim]Goodbye.[/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q", ":q"):
                console.print("[dim]Goodbye.[/dim]")
                break

            # ---- Input guardrails ---------------------------------------
            safe_input, input_blocked = await apply_input_guardrails(user_input)
            if input_blocked:
                console.print(
                    Panel(
                        Text(safe_input, style="bold yellow"),
                        title="[bold yellow]⚠ Input Check[/bold yellow]",
                        border_style="yellow",
                        padding=(0, 1),
                    )
                )
                console.print()
                continue

            # ---- Reset per-turn state ------------------------------------
            _fs.reset_turn_state()

            # ---- Run agent + ESC watcher concurrently -------------------
            logger.debug("User input: %s", user_input[:500])

            cancel_event: asyncio.Event = asyncio.Event()
            stream_task = asyncio.create_task(self._stream_response(user_input))
            watch_task = asyncio.create_task(_watch_for_esc(cancel_event))

            done, pending = await asyncio.wait(
                {stream_task, watch_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel whatever is still running
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # If the stream task itself raised (not cancelled), re-raise or log
            if stream_task in done:
                exc = stream_task.exception()
                if exc and not isinstance(exc, asyncio.CancelledError):
                    logger.error("Stream task raised: %s", exc)

            # Show interruption notice if ESC was pressed or command declined
            if cancel_event.is_set():
                if _fs.is_declined():
                    console.print(
                        "\n[dim](Command declined — returning to prompt)[/dim]\n"
                    )
                else:
                    console.print(
                        "\n[bold yellow]⊘ Interrupted[/bold yellow] "
                        "[dim](ESC pressed — returning to prompt)[/dim]\n"
                    )
