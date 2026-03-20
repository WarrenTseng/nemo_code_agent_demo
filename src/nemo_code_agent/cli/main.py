"""CLI entry point — ``code-agent`` command via Typer.

Usage examples::

    code-agent run
    code-agent run --session my-project-session
    code-agent run --checkpoint ./my_checkpoints.db

Environment variables (set in .env or shell):
    PLANNER_URL, PLANNER_MODEL, PLANNER_API_KEY
    CODER_URL, CODER_MODEL, CODER_API_KEY
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

app = typer.Typer(
    name="code-agent",
    help="Interactive multi-agent CLI coding assistant (Planner + Coder).",
    add_completion=False,
)
console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_env(env_file: Optional[Path]) -> None:
    """Load .env from the given path or fall back to the CWD .env."""
    if env_file and env_file.exists():
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)  # loads CWD/.env if present


def _check_required_env() -> None:
    """Fail fast with a clear message if required env vars are missing."""
    # API keys are optional — local servers don't need them.
    # The LLM factories substitute "none" when they are blank or absent.
    required = ["PLANNER_URL", "PLANNER_MODEL", "CODER_URL", "CODER_MODEL"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        console.print(
            f"[bold red]Error:[/bold red] Missing required environment variables:\n"
            + "\n".join(f"  • {v}" for v in missing)
            + "\n\nCopy [cyan].env.example[/cyan] to [cyan].env[/cyan] and fill in the values."
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("run")
def run(
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help=(
            "Project folder the agent works in. "
            "All file reads, writes, and shell commands are resolved relative to this path. "
            "Defaults to the current directory."
        ),
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help=(
            "Session ID for conversation persistence. "
            "Reusing the same ID resumes a previous session. "
            "Defaults to a new UUID each run."
        ),
    ),
    checkpoint_db: str = typer.Option(
        ".agent_logs/checkpoints.db",
        "--checkpoint",
        "-c",
        help="Path to the SQLite checkpoint database.",
        envvar="CODE_AGENT_CHECKPOINT_DB",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking",
        help="Enable extended reasoning mode for the Planner (Nemotron models).",
        hidden=True,
    ),
    show_thinking: bool = typer.Option(
        False,
        "--show-thinking",
        help="Stream the thinking process live in the terminal.",
        hidden=True,
    ),
    env_file: Optional[Path] = typer.Option(
        None,
        "--env-file",
        help="Path to a .env file (default: .env in current directory).",
    ),
    history_file: str = typer.Option(
        ".agent_logs/.repl_history",
        "--history",
        help="Path to the prompt_toolkit input history file.",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Skip the approval prompt for every bash command (requires explicit confirmation at startup).",
    ),
) -> None:
    """Start an interactive coding agent session."""
    _load_env(env_file)
    _check_required_env()

    if auto_approve:
        from rich.panel import Panel
        from rich.text import Text

        console.print()
        console.print(
            Panel(
                Text.assemble(
                    ("⚠  AUTO-APPROVE MODE\n\n", "bold red"),
                    ("With this flag, the agent will execute ALL bash commands and write ALL files "
                     "without asking for your approval first.\n\n", "yellow"),
                    ("This includes potentially destructive operations such as:\n", "yellow"),
                    ("  rm -rf, git reset --hard, DROP TABLE, curl | bash, overwriting any file, ...\n\n", "bold yellow"),
                    ("We suggest enabling this only in an isolated environment such as:\n", "white"),
                    ("  • A Docker container\n  • A sandbox VM\n  • A CI/CD pipeline with no sensitive data\n", "cyan"),
                ),
                title="[bold red]Security Warning[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
        )
        console.print()

        try:
            answer = input('\033[33mType "YES I UNDERSTAND" to enable auto-approve, or press Enter to cancel: \033[0m')
        except (KeyboardInterrupt, EOFError):
            answer = ""

        if answer.strip() != "YES I UNDERSTAND":
            console.print("[dim]Auto-approve cancelled. Starting with manual approval (safe mode).[/dim]")
            console.print()
            auto_approve = False
        else:
            from nemo_code_agent.tools import filesystem as _fs
            _fs.set_auto_approve(True)
            console.print("[bold yellow]⚠  Auto-approve ENABLED — all bash commands will run without confirmation.[/bold yellow]")
            console.print()

    # Resolve CODER_KNOWLEDGE_DIR to an absolute path before changing cwd,
    # so knowledge/static.md is always found relative to where the CLI was invoked.
    knowledge_dir = os.environ.get("CODER_KNOWLEDGE_DIR", "knowledge")
    if not os.path.isabs(knowledge_dir):
        os.environ["CODER_KNOWLEDGE_DIR"] = str(Path(knowledge_dir).resolve())

    # Change into the workspace directory so all relative paths in the tools
    # (read_file_tool, write_file_tool, execute_bash_tool) resolve there.
    if workspace:
        os.chdir(workspace)
        console.print(f"[dim]Workspace: {workspace}[/dim]")
    else:
        console.print(f"[dim]Workspace: {Path.cwd()}[/dim]")

    session_id = session or str(uuid.uuid4())
    planner_key = "****" if os.environ.get("PLANNER_API_KEY") else "(no key)"
    coder_key   = "****" if os.environ.get("CODER_API_KEY")   else "(no key)"
    console.print(
        f"[dim]Planner : {os.environ['PLANNER_MODEL']} @ {os.environ['PLANNER_URL']}  key={planner_key}[/dim]"
    )
    console.print(
        f"[dim]Coder   : {os.environ['CODER_MODEL']} @ {os.environ['CODER_URL']}  key={coder_key}[/dim]"
    )

    asyncio.run(_async_run(session_id, checkpoint_db, thinking, show_thinking, history_file))


async def _async_run(
    session_id: str,
    checkpoint_db: str,
    enable_thinking: bool,
    show_thinking: bool,
    history_file: str,
) -> None:
    """Async entry point — opens checkpointer and starts the REPL."""
    from nemo_code_agent.cli.repl import CodeAgentREPL
    from nemo_code_agent.workflow import create_agent

    # Ensure the history file directory exists
    Path(history_file).parent.mkdir(parents=True, exist_ok=True)

    async with create_agent(
        checkpoint_db=checkpoint_db,
        enable_thinking=enable_thinking,
    ) as agent:
        repl = CodeAgentREPL(
            agent,
            session_id=session_id,
            history_file=history_file,
            show_thinking=show_thinking,
        )
        await repl.run()


@app.command("sessions")
def sessions(
    checkpoint_db: str = typer.Option(
        ".agent_logs/checkpoints.db",
        "--checkpoint",
        "-c",
        help="Path to the SQLite checkpoint database.",
        envvar="CODE_AGENT_CHECKPOINT_DB",
    ),
) -> None:
    """List all saved session IDs in the checkpoint database."""
    from pathlib import Path as _Path  # noqa: PLC0415

    db = _Path(checkpoint_db)
    if not db.exists():
        console.print(f"[dim]No checkpoint database found at {db}[/dim]")
        raise typer.Exit()

    try:
        import sqlite3  # noqa: PLC0415

        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        conn.close()
    except Exception as exc:
        console.print(f"[bold red]Error reading checkpoint DB:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not rows:
        console.print("[dim]No sessions found.[/dim]")
    else:
        console.print(f"[bold]Saved sessions[/bold] ({db}):\n")
        for (thread_id,) in rows:
            console.print(f"  [cyan]{thread_id}[/cyan]")
        console.print(
            f"\n[dim]Resume with:  code-agent run --session <SESSION_ID>[/dim]"
        )


# ---------------------------------------------------------------------------
# Entrypoint guard (for direct python -m execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
