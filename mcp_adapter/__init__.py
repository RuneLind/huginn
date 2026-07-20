"""Package for the Knowledge API MCP adapter.

The runnable stdio entry point remains ``knowledge_api_mcp_adapter.py`` at the
repo root (user MCP configs reference it by path). This package holds the two
halves the entry file was split into:

- ``config``    — import-time config + feature detection (env-driven).
- ``formatting`` — pure markdown rendering for every tool's response.

The entry file re-exports the symbols tests reach for, so
``import knowledge_api_mcp_adapter as adapter`` continues to expose the full
surface with ``patch.object`` semantics intact.
"""
