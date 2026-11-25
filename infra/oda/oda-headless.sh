#!/bin/sh
set -eu

ODA_BIN="$(find /usr/bin -maxdepth 2 -type f -name ODAFileConverter 2>/dev/null | sort -V | tail -n 1 || true)"
if [ -z "$ODA_BIN" ]; then
    ODA_CMD="$(command -v ODAFileConverter || true)"
    [ -n "$ODA_CMD" ] && ODA_BIN="$(readlink -f "$ODA_CMD")"
fi
if [ -z "$ODA_BIN" ]; then
    echo "ODAFileConverter binary not found under /usr/bin" >&2
    exit 1
fi
ODA_DIR="$(dirname "$ODA_BIN")"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export QT_PLUGIN_PATH="$ODA_DIR/plugins"
export LD_LIBRARY_PATH="$ODA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ODA_DIR"
if [ -n "${CAD_XVFB_SERVER_NUM:-}" ]; then
    exec xvfb-run -n "$CAD_XVFB_SERVER_NUM" --server-args="-screen 0 1024x768x24" "$ODA_BIN" "$@"
fi
exec xvfb-run -a --server-args="-screen 0 1024x768x24" "$ODA_BIN" "$@"
