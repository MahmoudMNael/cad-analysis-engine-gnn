#!/bin/sh
set -eu

ODA_DIR="/usr/bin/ODAFileConverter_27.1.0.0"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export QT_PLUGIN_PATH="$ODA_DIR/plugins"
export LD_LIBRARY_PATH="$ODA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ODA_DIR"
exec xvfb-run -a --server-args="-screen 0 1024x768x24" "$ODA_DIR/ODAFileConverter" "$@"
