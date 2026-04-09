---
type: concept
title: "AI Agent Design Patterns"
aliases: ["AI Agent Design Patterns"]
created: 2026-04-06
updated: 2026-04-06
tags: [agents, architecture, design-patterns]
sources: ["[[Sources — AI General]]", "[[Sources — Claude Code]]"]
---

# AI Agent Design Patterns

Architectural patterns for building effective AI agents, synthesized from both Claude Code-specific sources and general AI agent literature.

## The Simple Loop (Claude Code's Approach)

[[Claude Code]]'s architecture: **call tools until none remain, then prompt the user.** Deliberately minimalist. Why it works:

- Simple tools (read, write, bash) have abundant training data
- Less complexity = fewer failure points
- Bash as universal adapter handles thousands of tasks
- Emergent complex behavior from simple primitives

Contrast with framework-based approaches (LangGraph, Pydantic AI) that require managing databases, defining tool schemas, wiring up RAG pipelines.

## The "New Way" vs. "Old Way"

Sources describe a split in agent building philosophy:

| Approach | Stack | Best For |
|----------|-------|----------|
| **Framework-based** | LangGraph, CrewAI, custom code | Production systems, custom integrations |
| **Harness-based** | Claude Code, Codex, Gemini CLI | Development tasks, prototyping, personal agents |
| **Hybrid** | Claude Code + MCP + custom tools | Best of both worlds |

The trend is toward harness-based: let the AI provider handle the agent loop, focus on tools and context.

## Key Design Principles

### From [[Andrej Karpathy]]'s 10 Principles
- Keep it simple, use natural language, give terminal access
- Let agents fail and retry (resilience > perfection)
- Make agents persistent and knowledge-accumulating
- Replace bespoke apps with agent-driven API glue

### From Google's AI Agent Paper
- Start with problems that have **known inputs and outputs**
- Focus on tasks that **reduce toil** and are **verifiable**
- Deliver **immediate value** before attempting complex autonomy

### From Anthropic's Long-Running Agent Blueprint
- Design for sustained operation, not one-shot tasks
- Build in heartbeat/health monitoring
- Plan for context refresh over long durations
- Use verification loops, not hope

## Agent Reliability Through Adversarial Patterns

A novel finding: agent reliability **explodes** when agents argue. Having a second agent challenge the first's output catches errors that self-review misses. This aligns with [[Verification-Driven Development]].

## Self-Learning Agent Architecture

Agents that improve through operation:
- Accumulate knowledge in persistent files
- Track corrections and patterns
- Apply learned patterns to future tasks
- See also: [[Skills System|self-improving skills]] in Claude Code

## Multi-Agent Orchestration

- Open-source orchestration layers (Paperclip, etc.) let you create teams from a single dashboard
- Agents run on **heartbeat systems** — spinning up, executing, reporting back
- Coordination is the hardest problem — most failures come from agent miscommunication, not individual agent errors

## The "90% Done" Consensus

Multiple sources note that by early 2026, standard AI coding workflows handle ~90% of development tasks. The remaining 10% — novel architecture decisions, complex debugging, cross-system integration — still requires human judgment.

## See also

- [[Andrej Karpathy]]
- [[Claude Code]]
- [[Subagents]]
- [[Agent Loops]]
- [[AI Coding Workflows]]
- [[Future of Software Engineering]]
