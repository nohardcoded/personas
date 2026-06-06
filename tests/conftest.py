"""Shared fixtures: a server module bound to a temp config + output dir (no model loading)."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def app_mod(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    cfg = tmp_path_factory.mktemp("cfg") / "config.yaml"
    cfg.write_text(
        f"output_dir: {out}\n"
        f"lock_file: {out / 'test.lock'}\n"
        "engines:\n"
        "  zimage:\n    type: zimage\n    label: Z-Image\n    model: /nonexistent/zimage\n"
        "  sdxl:\n    type: sdxl\n    label: SDXL\n    model: /nonexistent/sdxl\n"
    )
    os.environ["PERSONA_GEN_CONFIG"] = str(cfg)
    import persona_gen.server as server  # imported with the temp config in place

    return server


@pytest.fixture
def client(app_mod):
    from fastapi.testclient import TestClient

    # base_url must be a localhost host: the server's DNS-rebinding guard rejects the
    # default "testserver" Host header (a non-local name).
    return TestClient(app_mod.app, base_url="http://127.0.0.1")
