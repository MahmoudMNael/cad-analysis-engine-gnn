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

RUN pip install -e ".[dev]" \
    && mkdir -p /object-storage

EXPOSE 8000
