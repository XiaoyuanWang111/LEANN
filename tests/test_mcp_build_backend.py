"""leann_build must update an existing index with the backend it was built with.

`leann build` defaults to HNSW and the MCP tool used to pass `--backend-name=ivf` whenever the
caller named no backend. On an index made by `leann build my-project --docs ...` (the MCP
README's quick start), adding a file then failed with "Index was built with backend 'hnsw',
cannot update with 'ivf'", and editing one started a full rebuild with the IVF backend.
"""

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

module_path = Path(__file__).resolve().parents[1] / "packages/leann-core/src/leann/mcp.py"
mcp = ModuleType("leann_mcp_build_backend_test")
SourceFileLoader(mcp.__name__, str(module_path)).exec_module(mcp)


def _backend_passed_to_cli(tmp_path, monkeypatch, arguments, existing_backend=None):
    if existing_backend is not None:
        index_dir = tmp_path / ".leann" / "indexes" / "my-project"
        index_dir.mkdir(parents=True)
        (index_dir / "documents.leann.meta.json").write_text(
            json.dumps(
                {
                    "backend_name": existing_backend,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_mode": "sentence-transformers",
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(mcp, "_base_dir", str(tmp_path))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(mcp.subprocess, "run", fake_run)
    mcp.handle_build(1, {"docs": ["src"], **arguments})

    (command,) = calls
    flags = [arg for arg in command if arg.startswith("--backend-name=")]
    assert len(flags) == 1
    return flags[0].split("=", 1)[1]


@pytest.mark.parametrize("existing_backend", ["hnsw", "ivf"])
def test_an_existing_index_keeps_its_backend(tmp_path, monkeypatch, existing_backend):
    backend = _backend_passed_to_cli(
        tmp_path, monkeypatch, {"index_name": "my-project"}, existing_backend
    )
    assert backend == existing_backend


def test_a_new_index_still_defaults_to_ivf(tmp_path, monkeypatch):
    assert _backend_passed_to_cli(tmp_path, monkeypatch, {"index_name": "my-project"}) == "ivf"


def test_an_explicit_backend_still_wins(tmp_path, monkeypatch):
    backend = _backend_passed_to_cli(
        tmp_path, monkeypatch, {"index_name": "my-project", "backend_name": "ivf"}, "hnsw"
    )
    assert backend == "ivf"
