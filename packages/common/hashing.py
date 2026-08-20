"""Canonical hashing for stable geometry IDs and job reproducibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(
    kind: str,
    rounded_coordinates: Sequence[Any],
    relevant_attributes: Mapping[str, Any],
) -> str:
    """Single canonical geometry hash. Do not duplicate this logic elsewhere."""
    payload = {
        "kind": kind,
        "coords": _to_jsonable(rounded_coordinates),
        "attrs": _to_jsonable(dict(relevant_attributes)),
    }
    return sha256_hex(canonical_json(payload).encode("utf-8"))


def configuration_hash(config: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(config)).encode("utf-8"))


def engine_sha_from_sources(root: Path) -> str:
    digest = hashlib.sha256()
    roots = [root / "packages", root / "apps"]
    files: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        files.extend(sorted(p for p in base.rglob("*.py") if p.is_file()))
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return float(value)
    return value
