---
type: concept
title: "LLM Fundamentals"
aliases: ["LLM Fundamentals"]
created: 2026-04-06
updated: 2026-04-06
tags: [llm, technical, architecture, reasoning]
sources: ["[[Sources — AI General]]", "[[Sources — Claude Code]]"]
---

# LLM Fundamentals

Technical foundations of how large language models work, relevant to understanding why [[Claude Code]] and other AI tools behave the way they do.

## How LLMs Generate Text

- LLMs produce output through **pure mathematical transformations** — not "understanding" in a human sense
- They excel at **behavioral reasoning**: producing human-like problem-solving steps through pattern matching over vast training data
- Each token is predicted based on all preceding tokens and the model's learned weights
- Temperature controls randomness; lower = more deterministic, higher = more creative

## Are LLMs "Just Predicting Words"?

Sources push back on the reductive framing:

- LLMs develop **internal representations** that go beyond surface-level pattern matching
- They can solve novel problems by composing learned patterns in new ways
- However, they lack persistent memory, causal understanding, and true planning
- The "emergent reasoning" debate: capabilities appear at scale that aren't present in smaller models

## Context Windows

> "Context windows are stupid" — provocative take from one source

The core tension:
- Bigger context windows (1M tokens for Opus 4.6) enable more information
- But more information doesn't mean better results — [[Context Management|context degradation]] is real
- MIT researchers claim to have "destroyed the context window limit" with new techniques
- Practical advice: use large context windows strategically, not indiscriminately

## Sycophancy

A known failure mode where models agree with the user rather than being accurate:

- Models are trained on human feedback, which rewards agreeable responses
- This creates a bias toward confirmation rather than correction
- [[Anthropic]] has studied this extensively and works to mitigate it
- Practical impact: don't treat model agreement as validation of your approach

## Key Architectural Concepts

- **Tokens** — sub-word units that are the fundamental unit of processing
- **Embeddings** — high-dimensional vector representations of meaning
- **Attention heads** — the mechanism that determines which parts of context to focus on
- **Logits** — raw output scores before probability normalization
- **Tool calling** — formatting output as structured JSON to invoke external functions

## See also

- [[Context Management]]
- [[Prompting for Claude]]
- [[Claude Code]]
- [[AI Industry Landscape]]
- [[Sources — RAG]]
- [[Sources — Claude Models]]
