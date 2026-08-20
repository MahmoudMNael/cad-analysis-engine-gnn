from __future__ import annotations

import os
from pathlib import Path

import pytest

from packages.common.configuration import reset_settings_cache

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_settings():
    reset_settings_cache()
    yield
    reset_settings_cache()


def integration_enabled() -> bool:
    return os.environ.get("CAD_RUN_INTEGRATION") == "1"
