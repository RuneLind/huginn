---
type: entity
title: "NotebookLM"
aliases: ["NotebookLM"]
created: 2026-04-06
updated: 2026-04-06
tags: [product, google, ai-tool, research]
sources: ["[[Sources — AI General]]", "[[Sources — Claude Code]]"]
---

# NotebookLM

Google's AI research tool. Acts as a pre-built RAG system: ingestion, indexing, analysis, and deliverables — all free.

## Key Features

- **Source-grounded AI** — always cites its sources, reducing hallucination
- **Audio overviews** — generates podcast-style discussions of your sources
- **Data Tables** — structured data analysis (launched December 2025 for Pro/Ultra users)
- **Gemini integration** — notebooks attach as context in the Gemini app, enabling a research-to-creation pipeline

## As a Claude Code Companion

Multiple sources describe a "cheat code" workflow:

1. Feed sources into NotebookLM for free RAG-based analysis
2. Use NotebookLM's output as structured input for [[Claude Code]]
3. Claude Code builds on top of NotebookLM's synthesis

A custom [[Model Context Protocol|MCP]] server for NotebookLM was built with 31 tools using reverse-engineered Google RPC calls — faster than browser automation.

## Use Cases from Sources

- Content strategy analysis
- Source quality evaluation
- Research synthesis
- Meeting note processing
- Educational content creation
- Challenge assumptions and identify gaps

## See also

- [[Claude Code]]
- [[Model Context Protocol]]
- [[AI Industry Landscape]]
