"""Manual pipeline stage timing helper. Not a CI gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.helpers.pipeline import run_dxf


def main() -> None:
    parser = argparse.ArgumentParser(description="Print pipeline stage_timings for a DXF.")
    parser.add_argument("dxf", type=Path)
    args = parser.parse_args()
    result = run_dxf(args.dxf)
    timings = result.meta.get("stage_timings") or {}
    print(json.dumps(timings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
