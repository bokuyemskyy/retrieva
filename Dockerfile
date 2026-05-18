FROM ghcr.io/astral-sh/uv:0.11.14-python3.12-trixie-slim AS builder
WORKDIR /app

# Only copy uv.lock and not pyproject.toml
# This ensures hermiticity of the build
# And prevents docker image invalidation in case non-dependency changes
# are made to pyproject.toml
COPY uv.lock /app

# Install dependencies
# virtual env is created in "/app/.venv" directory
RUN uv init --name retrieva && uv sync --no-dev --frozen



FROM python:3.12-slim-trixie AS runner
RUN apt-get update
RUN apt-get -y install tesseract-ocr

COPY --from=builder /app/.venv /app/.venv
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/.venv/lib/python3.12/site-packages

COPY app /app/app
COPY core /app/core
COPY scripts /app/scripts

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app/app.py"]