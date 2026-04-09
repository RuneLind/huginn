---
type: entity
title: "OpenClaw"
aliases: ["OpenClaw"]
created: 2026-04-06
updated: 2026-04-06
tags: [product, ai, coding-tool, open-source]
sources: ["[[Sources — Claude Code]]"]
---

# OpenClaw

An open-source AI coding agent created by Peter Steinberger. Positioned as an alternative to [[Claude Code]], using the same underlying Claude models via API.

## Key Features

- Open-source codebase
- Telegram integration for mobile control
- Uses Claude API directly (pay per token)
- Community-driven development

## Claude Code vs. OpenClaw

The comparison is a recurring topic across multiple sources:

### Cost

- OpenClaw uses API pricing (pay per token) — can be cheaper for light usage
- Claude Code's $200/month subscription gives ~$5,000 worth of API usage — far cheaper for heavy users
- One source reports **94 commits in a single day, 7 PRs in 30 minutes** using OpenClaw + Codex swarm at ~$100-190/month

### Capabilities

- Claude Code has native [[Skills System|skills]], [[Hooks and Automation|hooks]], [[Subagents]], and plugin ecosystem
- OpenClaw is more bare-bones but fully customizable
- Claude Code's `claude -p` command provides a programmable alternative that largely replaces OpenClaw's use case

### The Ban

Anthropic banned OpenClaw from using the Claude subscription (Max plan) tokens, restricting it to API-only access. This made it significantly more expensive for heavy users and pushed many users back to Claude Code.

### Community Consensus

Multiple sources converge on: Claude Code has overtaken OpenClaw for most users due to the subscription pricing advantage, native integrations, and the `claude -p` programmable mode. OpenClaw remains relevant for users who want full control and are willing to pay API rates.

## Agent Swarm Pattern

A notable use pattern: running OpenClaw alongside Codex and Claude Code as a multi-agent swarm, each handling different aspects of development simultaneously.

## See also

- [[Claude Code]]
- [[Anthropic]]
- [[AI Industry Landscape]]
- [[Sources — OpenClaw]]
