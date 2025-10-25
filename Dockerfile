# --- Stage 1: get uv binaries ---
FROM ghcr.io/astral-sh/uv:latest AS uv

# --- Stage 2: main Drake environment ---
FROM robotlocomotion/drake:noble

# Copy uv + uvx binaries from stage 1
COPY --from=uv /uv /uvx /usr/local/bin/

# Create working directory
WORKDIR /drake

# System dependencies
RUN apt-get update && apt-get install -y \
    python3-dev cmake build-essential git vim wget curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /app

# Dependency installation layer (cached)
# Mount pyproject.toml & uv.lock only for dependency caching
RUN --mount=type=cache,target=/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,readonly \
    --mount=type=bind,source=uv.lock,target=uv.lock,readonly \
    uv sync --locked --no-install-project

# Add full source code
ADD . /drake

# Final sync to install your project itself
RUN --mount=type=cache,target=/.cache/uv \
        uv sync --locked

# Activate .venv for runtime
ENV PATH="/drake/.venv/bin:$PATH"

RUN git config --global --add safe.directory /drake

CMD ["bash"]
