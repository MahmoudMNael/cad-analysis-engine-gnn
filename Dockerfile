FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAD_OBJECT_STORE_DIR=/object-storage \
    CAD_ODA_CONVERTER_PATH=/usr/local/bin/oda-headless.sh

ARG ODA_DEB_URL=https://www.opendesign.com/guestfiles/get?filename=ODAFileConverter_QT6_lnxX64_8.3dll_27.1.deb

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        xvfb \
        xauth \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-xkb1 \
        libxcb-xinerama0 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libxext6 \
        libxrender1 \
        libxi6 \
        libsm6 \
        libice6 \
        libegl1 \
        libfreetype6 \
        libgl1 \
        libglib2.0-0 \
        libdbus-1-3 \
        libfontconfig1 \
        libxcb-util1 \
    && ln -sf /usr/lib/x86_64-linux-gnu/libxcb-util.so.1 /usr/lib/x86_64-linux-gnu/libxcb-util.so.0 \
    && wget -q -O /tmp/oda-file-converter.deb "$ODA_DEB_URL" \
    && apt-get install -y /tmp/oda-file-converter.deb \
    && rm /tmp/oda-file-converter.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY infra/oda/oda-headless.sh /usr/local/bin/oda-headless.sh
RUN chmod +x /usr/local/bin/oda-headless.sh

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
