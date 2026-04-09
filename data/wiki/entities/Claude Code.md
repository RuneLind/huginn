---
type: entity
title: "Claude Code"
aliases: ["Claude Code"]
created: 2026-04-06
updated: 2026-04-06
tags: [product, ai, coding-tool]
sources: ["[[Sources — Claude Code]]"]
---

# Claude Code

An AI coding agent built by [[Anthropic]], designed as a CLI-first tool that operates directly in your terminal. Created by [[Boris Cherny]].

## Architecture

Claude Code uses a simple master loop: **call tools until none remain, then prompt the user.** This is deliberately minimalist compared to complex multi-layer agent designs.

### Core Tools

Only four fundamental tools:
- **Read** — read files
- **Write/Edit** — create and modify files
- **Grep/Glob** — search files and content
- **Bash** — execute shell commands (the "universal adapter")

Everything else is built on top through [[Model Context Protocol|MCP]], [[Skills System|skills]], and [[Subagents]].

## Extension Layers

1. **[[Model Context Protocol|MCP]]** — connect to external systems and data sources
2. **[[Skills System|Skills]]** — reusable procedural knowledge and workflows
3. **[[Subagents]]** — specialized agents with isolated context
4. **[[Hooks and Automation|Hooks]]** — deterministic lifecycle automation
5. **Plugins** — bundled packages of skills, hooks, agents, and MCP configs

## Key Features

- **[[Context Management]]** with automatic compaction and progressive disclosure
- **[[Agent Loops|Background agents]]** for parallel task execution
- **[[Agent Loops|Scheduled agents]]** (triggers) for cron-based automation
- **LSP support** — IDE-like features (jump to definition, find references, hover type info)
- **Chrome integration** — browser automation for UI testing
- **Ultra Think mode** — extended reasoning for complex tasks
- **Plugin marketplace** — 42+ official plugins
- **Channels** — extensible communication layer for external integrations
- **Session management** — `/rename`, `/resume` for managing parallel work

## Pricing

- **$200/month subscription** (Claude Max) — gives approximately $5,000 worth of API usage
- API access available separately at per-token rates
- The subscription is heavily subsidized relative to raw API costs

## Platform Evolution

Sources describe Claude Code evolving from a single coding agent into a **multi-agent orchestration platform** — an "agent operating system" with coordinated updates across browser, Slack, terminal, and mobile surfaces.

## Notable Event

In early April 2026, Anthropic accidentally leaked Claude Code's entire source code while shipping an April Fools' Tamagotchi feature update. Analysis of the leak confirmed the simple architecture described in official communications.

## Competitors

- **[[OpenClaw]]** — open-source alternative, uses same underlying models
- **Cursor** — IDE-based AI coding
- **GitHub Copilot** — inline code completion and chat
- **OpenAI Codex** — OpenAI's coding agent
- **Gemini CLI** — Google's command-line AI tool

## See also

- [[Anthropic]]
- [[Boris Cherny]]
- [[Context Management]]
- [[Skills System]]
- [[Subagents]]
- [[Model Context Protocol]]
- [[CLAUDE.md Configuration]]
