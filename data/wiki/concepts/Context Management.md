---
type: concept
title: "Context Management"
aliases: ["Context Management"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, architecture, performance]
sources: ["[[Sources — Claude Code]]"]
---

# Context Management

The single most important concept in effective [[Claude Code]] use. Context fullness directly degrades model intelligence — as the context window fills, the model loses ability to follow instructions, generates more errors, and produces lower-quality output.

## The Problem

- Claude prioritizes the **beginning and end** of context; content in the middle gets lost (the "lost in the middle" problem)
- Every tool call, file read, and MCP response adds tokens to the context
- Long conversations compound: each new operation adds tokens, degrading quality over time
- At high context utilization, accuracy can drop dramatically

## Core Principle

> **Keep main context below 50% utilization.** This is the single most impactful practice.

## Strategies

### Subagent Delegation

Delegate finite, self-contained tasks to [[Subagents]]. Each subagent gets its own isolated context window. Only the result comes back to the main conversation. This is the primary mechanism for keeping context clean.

### Context Compaction

Claude automatically compacts context when approaching limits — summarizing older messages, removing stale tool call results, preserving conversation flow. This happens transparently but can be triggered manually with `/compact`.

> [!tip]
> Use `/clear` between unrelated tasks. Start a new session for a fresh context rather than continuing a long one.

### Progressive Disclosure

Load information on-demand rather than upfront. This applies to:

- **[[Skills System|Skills]]**: only metadata is shown at runtime; full instructions are loaded when the skill is triggered
- **[[Model Context Protocol|MCP]] tools**: dynamic tool discovery loads only 3-5 relevant tools per prompt instead of all available tools
- **Documentation**: reference files are loaded only when needed via `@imports`

Progressive disclosure has been shown to reduce context usage by **85%** and improve accuracy from 49% to 74%+ on evaluations.

### Context Engineering

The meta-skill of managing what information the model sees and when. Five core sub-skills identified across sources:

1. **Mapping** — break implementation into properly-sized tasks
2. **Agentic workflows** — use subagents for parallel, isolated tasks
3. **Context engineering** — balance information availability vs. dilution
4. **Harness engineering** — master your tools deeply (hooks, skills, MCP)
5. **Model intuition** — recognize LLM patterns and failure modes

### Session Management

- Use `/rename` and `/resume` to manage multiple sessions
- Plan in one session, execute in another (fresh context)
- `/clear` between unrelated tasks within a session

## Anti-Patterns

- Dumping entire codebases into context upfront
- Running many MCP tools that return large outputs
- Continuing conversations long past when they should have been cleared
- Loading all skills/tools at startup rather than on-demand

## See also

- [[Subagents]]
- [[Skills System]]
- [[Model Context Protocol]]
- [[Claude Code]]
- [[Prompting for Claude]]
- [[LLM Fundamentals]]
- [[Sources — RAG]]
