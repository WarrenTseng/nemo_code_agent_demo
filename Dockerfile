# NeMo Code Agent — Docker image
# Base: NVIDIA PyTorch 26.02 (ships Python 3.12, CUDA, cuDNN)
# Tag:  warrents/nemo-code-agent-demo:0.0.1
#
# Build:
#   docker build -t warrents/nemo-code-agent-demo:0.0.1 .
#
# Run (interactive):
#   docker run -it --rm \
#     --env-file .env \
#     warrents/nemo-code-agent-demo:0.0.1
#
# Run with a host workspace folder mounted:
#   docker run -it --rm \
#     --env-file .env \
#     -v $(pwd)/myproject:/workspace/project \
#     warrents/nemo-code-agent-demo:0.0.1 --workspace /workspace/project
#
# Run with GPU (if needed by local vLLM inside the container):
#   docker run -it --rm --gpus all --env-file .env \
#     warrents/nemo-code-agent-demo:0.0.1

FROM nvcr.io/nvidia/pytorch:26.02-py3

LABEL maintainer="WarrenTseng" \
      version="0.0.1" \
      description="NeMo Code Agent — interactive multi-agent CLI coding assistant"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

WORKDIR /workspace/nemo_code_agent

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
# Copy only the packaging files first so this layer is cached until they change
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package with NAT + Guardrails extras.
# --prefer-binary avoids source builds (prevents pandas/NumPy 2.x issues).
# NAT install is attempted; if it fails due to Python version it is non-fatal.
RUN pip install --prefer-binary -e ".[guardrails]" && \
    pip install --prefer-binary -e ".[nat]" || \
    echo "[docker] NAT install skipped (not compatible with this Python version)"

# ---------------------------------------------------------------------------
# Copy remaining project files
# ---------------------------------------------------------------------------
COPY guardrails/ ./guardrails/
COPY configs/    ./configs/
COPY .env.example ./

# Runtime directories for logs and SQLite checkpoints
RUN mkdir -p .agent_logs

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
# .env is NOT baked in — pass secrets at runtime via --env-file or -e flags.
# Extra arguments to `docker run` are forwarded to `code-agent run`:
#   docker run ... warrents/nemo-code-agent-demo:0.0.1 --session my-session

ENTRYPOINT ["code-agent", "run"]
