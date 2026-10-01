# Reproducible pyspi environment built from the committed uv.lock.
#
# linux/amd64 is pinned deliberately: dtaidistance publishes no linux-aarch64
# wheel, so an arm64 build would fall back to its sdist and require a C
# toolchain. Every dependency in uv.lock has a manylinux x86_64 wheel for
# CPython 3.12, so no compiler is installed here.
FROM --platform=linux/amd64 python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Keep the environment outside the workdir so it can never be shadowed by a
# host .venv, and so the image works with `python` straight off PATH.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /pyspi

# Dependencies first: this layer is only invalidated when the lock or the
# project metadata changes, not when library source changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Then the package itself.
COPY pyspi ./pyspi
RUN uv sync --frozen

CMD ["python"]
