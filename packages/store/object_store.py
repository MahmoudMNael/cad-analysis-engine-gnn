"""Local directory object storage. Keys are generated; never user-controlled paths."""

from __future__ import annotations

import re
from pathlib import Path

from packages.common.errors import StorageError
from packages.common.hashing import sha256_file, sha256_hex

_SAFE_KEY = re.compile(r"^[A-Za-z0-9._/-]+$")
REQUIRED_LAYOUT_KINDS = (
    "source",
    "document",
    "plan_graph",
    "faces",
    "layout",
    "meta",
)


class ObjectStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        if not _SAFE_KEY.match(key) or ".." in key.split("/"):
            raise StorageError("invalid object key")
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root)):
            raise StorageError("object key escaped storage root")
        return path

    def put_bytes(self, key: str, data: bytes) -> tuple[str, str]:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.uri_for(key), sha256_hex(data)

    def put_file(self, key: str, source: Path) -> tuple[str, str]:
        data = Path(source).read_bytes()
        return self.put_bytes(key, data)

    def get_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.is_file():
            raise StorageError(f"object not found: {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def uri_for(self, key: str) -> str:
        return f"file://{self._resolve(key)}"

    def key_for_job(self, job_id: str, filename: str) -> str:
        return f"jobs/{job_id}/{filename}"

    def healthcheck(self) -> bool:
        probe = self.root / ".health"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            return probe.read_text(encoding="utf-8") == "ok"
        except OSError:
            return False

    def snapshot_complete(self, job_id: str) -> bool:
        names = {
            "source": "source.dxf",
            "document": "document.json",
            "plan_graph": "plan_graph.json",
            "faces": "faces.parquet",
            "layout": "layout.json",
            "meta": "meta.json",
        }
        return all(self.exists(self.key_for_job(job_id, name)) for name in names.values())

    def digest_file(self, key: str) -> str:
        return sha256_file(self._resolve(key))
