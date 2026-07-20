"""Mechanical tool-surface diff for the MCP adapter split (PR E).

PR E splits ``knowledge_api_mcp_adapter.py`` into a package. The split must be
byte-compatible on the MCP tool surface: every tool name, description, and
parameter schema must be identical to the pre-split (origin/main) module. This
test registers the tools from *both* the current module and the origin/main
version of the file (loaded in isolation from git) under an identical
environment, then asserts the exposed tool set is identical.

If ``git show origin/main:knowledge_api_mcp_adapter.py`` is unavailable (e.g. a
shallow checkout in CI), the origin comparison is skipped and the test only
asserts the current surface is internally well-formed.

Run standalone for a human-readable diff:  python tests/test_mcp_tool_surface.py
"""
import asyncio
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Feature flags are read from the environment at import time. Pin a superset so
# every conditional tool (notion + graph) is registered and compared.
_ENV = {
    "KNOWLEDGE_COLLECTIONS": "",  # None → all features permitted (notion on)
    "KNOWLEDGE_API_URL": "http://localhost:8321",
    "KNOWLEDGE_DESCRIPTION": "surface-diff fixture",
    # _has_graph only checks Path.exists() — point it at any real file so the
    # two graph tools register and get schema-compared.
    "KNOWLEDGE_GRAPH_PATH": str(REPO_ROOT / "README.md"),
    "JIRA_GRAPH_PATH": "",
}


def _tool_surface(mcp) -> dict:
    """Extract {name: {description, parameters}} from a FastMCP instance."""
    tools = asyncio.run(mcp.list_tools())
    surface = {}
    for t in tools:
        surface[t.name] = {
            "description": t.description,
            "inputSchema": t.inputSchema,
        }
    return surface


def _reload_adapter():
    """Reload config + entry under the *current* environment."""
    config = importlib.import_module("mcp_adapter.config")
    importlib.reload(config)
    mod = importlib.import_module("knowledge_api_mcp_adapter")
    return importlib.reload(mod)


def _current_surface() -> dict:
    try:
        with _patched_env():
            # config caches feature flags at import time; reload it under the
            # patched env first so the entry module's re-import picks up fresh
            # flags (otherwise a prior real-env import shadows the graph tools).
            mod = _reload_adapter()
            return _tool_surface(mod.mcp)
    finally:
        # Return the modules to the real-env baseline so we don't leak the
        # fixture's feature flags into later tests in the same process.
        _reload_adapter()


def _origin_surface() -> dict | None:
    """Load origin/main's single-file adapter in isolation and register its tools."""
    try:
        src = subprocess.check_output(
            ["git", "show", "origin/main:knowledge_api_mcp_adapter.py"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return None

    with _patched_env():
        mod = types.ModuleType("_origin_mcp_adapter")
        mod.__file__ = str(REPO_ROOT / "knowledge_api_mcp_adapter.py")
        # The origin module does `if __name__ == "__main__"` — give it a
        # non-main name so it does not start the stdio server.
        sys.modules["_origin_mcp_adapter"] = mod
        try:
            exec(compile(src, mod.__file__, "exec"), mod.__dict__)
            return _tool_surface(mod.mcp)
        finally:
            sys.modules.pop("_origin_mcp_adapter", None)


class _patched_env:
    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in _ENV}
        os.environ.update(_ENV)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


EXPECTED_TOOLS = {
    "search_knowledge",
    "get_document",
    "list_collections",
    "list_tags",
    "get_notion_page",
    "get_graph_node",
    "get_graph_subtree",
}


def test_current_surface_registers_expected_tools():
    surface = _current_surface()
    assert set(surface) == EXPECTED_TOOLS


def test_surface_matches_origin_main():
    origin = _origin_surface()
    if origin is None:
        pytest.skip("origin/main:knowledge_api_mcp_adapter.py not available")
    current = _current_surface()
    assert set(current) == set(origin), (
        f"tool names differ: only-current={set(current) - set(origin)} "
        f"only-origin={set(origin) - set(current)}"
    )
    for name in origin:
        assert current[name]["description"] == origin[name]["description"], f"description differs for {name}"
        assert current[name]["inputSchema"] == origin[name]["inputSchema"], f"inputSchema differs for {name}"


if __name__ == "__main__":
    origin = _origin_surface()
    current = _current_surface()
    if origin is None:
        print("origin/main version unavailable — printing current surface only")
        print(json.dumps(current, indent=2, sort_keys=True))
        raise SystemExit(0)
    names_diff = set(current) ^ set(origin)
    print(f"tool names: current={sorted(current)}")
    print(f"name symmetric-diff: {names_diff or 'EMPTY'}")
    any_diff = bool(names_diff)
    for name in sorted(set(origin) & set(current)):
        if current[name]["description"] != origin[name]["description"]:
            any_diff = True
            print(f"[DESC DIFF] {name}")
        if current[name]["inputSchema"] != origin[name]["inputSchema"]:
            any_diff = True
            print(f"[SCHEMA DIFF] {name}")
            print("  origin :", json.dumps(origin[name]["inputSchema"], sort_keys=True))
            print("  current:", json.dumps(current[name]["inputSchema"], sort_keys=True))
    print("SURFACE DIFF:", "NON-EMPTY" if any_diff else "EMPTY")
    raise SystemExit(1 if any_diff else 0)
