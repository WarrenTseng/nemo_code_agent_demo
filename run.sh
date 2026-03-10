#!/usr/bin/env bash
# run.sh — one-command launcher for the NeMo Code Agent
#
# Always launches the interactive REPL via: code-agent run [OPTIONS]
# NAT (nvidia-nat) is installed for its LLM provisioning library but the
# interactive CLI always uses code-agent run directly.
#
# Usage:
#   ./run.sh                                     # start a new session in CWD
#   ./run.sh -w ~/projects/myapp                 # work inside a specific folder
#   ./run.sh --workspace ~/projects/myapp        # same, long form
#   ./run.sh --session my-project                # resume a named session
#   ./run.sh -w ~/projects/myapp -s my-project   # workspace + named session
#   ./run.sh --thinking                          # enable Nemotron reasoning mode
#   ./run.sh --checkpoint ./my.db                # custom SQLite checkpoint path
#   ./run.sh --help                              # show all options

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve the project root (directory containing this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[run.sh]${RESET} $*"; }
success() { echo -e "${GREEN}[run.sh]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[run.sh]${RESET} $*"; }
error()   { echo -e "${RED}[run.sh] ERROR:${RESET} $*" >&2; }

# ---------------------------------------------------------------------------
# 1. Check Python version (>= 3.10)
# ---------------------------------------------------------------------------
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    error "Python 3.10 or later is required but was not found."
    error "Install it from https://python.org and re-run this script."
    exit 1
fi

info "Using Python: $PYTHON_BIN ($(${PYTHON_BIN} --version))"

# ---------------------------------------------------------------------------
# 2. Create virtual environment if it does not exist
# ---------------------------------------------------------------------------
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment at .venv ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created."
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# 3. Install / upgrade the package when pyproject.toml is newer than the
#    sentinel file (.venv/.install_ok), or when the venv was just created.
# ---------------------------------------------------------------------------
SENTINEL="$VENV_DIR/.install_ok"

needs_install=false
if [ ! -f "$SENTINEL" ]; then
    needs_install=true
elif [ "$SCRIPT_DIR/pyproject.toml" -nt "$SENTINEL" ]; then
    warn "pyproject.toml has changed — reinstalling package."
    needs_install=true
fi

if $needs_install; then
    pip install --quiet --upgrade pip

    # Install guardrails (always — pandas>=2.2 avoids NumPy 2.x build failures)
    info "Installing nemo-code-agent + guardrails ..."
    pip install --quiet --prefer-binary -e "$SCRIPT_DIR[guardrails]"

    # Install NAT separately — nvidia-nat requires Python <3.14, so this may
    # legitimately fail on newer Python versions.  Treat it as optional.
    # uv is used as a fallback because nvidia-nat's dependency graph is too
    # complex for pip's default resolver.
    info "Installing NAT support (optional — requires Python <3.14) ..."
    if pip install --quiet --prefer-binary -e "$SCRIPT_DIR[nat]" 2>/dev/null; then
        success "NAT installed."
    elif command -v uv &>/dev/null && uv pip install --quiet -e "$SCRIPT_DIR[nat]" 2>/dev/null; then
        success "NAT installed (via uv)."
    else
        warn "NAT install skipped (nvidia-nat does not support this Python version)."
        warn "The agent will run via code-agent run instead of nat run."
    fi

    touch "$SENTINEL"
    success "Installation complete."
fi

# ---------------------------------------------------------------------------
# 4. Verify .env exists
# ---------------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    warn ".env file not found."
    warn "Copying .env.example → .env  (you must fill in your API keys)"
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo ""
    echo -e "${YELLOW}Please edit .env and set your API keys, then re-run this script:${RESET}"
    echo "  ${CYAN}PLANNER_URL${RESET}  — NIM inference endpoint"
    echo "  ${CYAN}PLANNER_MODEL${RESET}— Planner model name"
    echo "  ${CYAN}PLANNER_API_KEY${RESET} — NIM API key (nvapi-...)"
    echo "  ${CYAN}CODER_URL${RESET}   — Coder endpoint (NIM or local vLLM)"
    echo "  ${CYAN}CODER_MODEL${RESET} — Coder model name"
    echo "  ${CYAN}CODER_API_KEY${RESET} — Coder API key (or 'none' for local vLLM)"
    echo ""
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Ensure the log directory exists
# ---------------------------------------------------------------------------
mkdir -p "$SCRIPT_DIR/.agent_logs"

# ---------------------------------------------------------------------------
# 6. Launch the agent
# ---------------------------------------------------------------------------
echo ""
info "Starting NeMo Code Agent..."
echo ""

if command -v nat &>/dev/null; then
    info "NAT library is available (LLM provisioning active)."
fi

exec code-agent run "$@"
