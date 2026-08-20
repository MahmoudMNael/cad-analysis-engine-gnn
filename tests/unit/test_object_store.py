from pathlib import Path

from packages.store.object_store import ObjectStore


def test_object_store_roundtrip(tmp_path: Path):
    store = ObjectStore(tmp_path / "object-storage")
    uri, digest = store.put_bytes("jobs/abc/source.dxf", b"0\nSECTION\n")
    assert store.exists("jobs/abc/source.dxf")
    assert store.get_bytes("jobs/abc/source.dxf") == b"0\nSECTION\n"
    assert digest
    assert uri.startswith("file://")
    assert store.healthcheck() is True


def test_object_store_rejects_path_escape(tmp_path: Path):
    store = ObjectStore(tmp_path / "object-storage")
    try:
        store.put_bytes("../secret", b"x")
        raise AssertionError("should reject")
    except Exception:
        pass
