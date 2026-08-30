FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAD_OBJECT_STORE_DIR=/object-storage

COPY pyproject.toml /app/pyproject.toml
COPY apps /app/apps
COPY packages /app/packages
COPY tests /app/tests
COPY main.py /app/main.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --retries 20 -e ".[dev]" \
    && pip install --default-timeout=1000 --retries 20 --index-url https://download.pytorch.org/whl/cpu torch
# 2) tests last (changes often, doesn't force torch reinstall if you split RUN)
COPY tests /app/tests
RUN mkdir -p /object-storage


EXPOSE 8000
