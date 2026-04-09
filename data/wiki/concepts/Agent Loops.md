---
type: concept
title: "Agent Loops"
aliases: ["Agent Loops"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, agents, automation, background]
sources: ["[[Sources — Claude Code]]"]
---

# Agent Loops

Agent loops extend [[Claude Code]] from an interactive assistant into an autonomous platform capable of running tasks in the background, on schedule, and across multiple environments.

## Core Architecture

Claude Code's master loop is deceptively simple: **call tools until none remain, then prompt the user.** This simple loop, combined with basic tools (read, write, edit, bash), produces emergent complex behavior.

## Background Agents

Background agents execute tasks silently without blocking the main session:

- Spawn with `claude --background` or via the Agent tool
- Each agent gets its own isolated context window
- Multiple background agents can run in parallel
- Results are returned when complete

Use cases:
- Running test suites while continuing development
- Parallel code reviews across multiple files
- Research tasks that feed into later decisions

## Ralph Loop

A pattern described by [[Anthropic]] for continuous autonomous operation:

1. Claude Code executes a task
2. A stop [[Hooks and Automation|hook]] triggers verification
3. If verification fails, a new Claude Code session is spawned to fix the issue
4. The cycle repeats until verification passes

Named after the Ralph Wiggum plugin used for visual verification. Enables Claude Code to run **24/7** on long-running tasks.

## Scheduled Agents (Triggers)

Claude Code can run on a cron schedule:

- Define triggers with cron expressions
- Agent executes in a sandboxed environment
- Results are available for review
- Useful for: daily code reviews, dependency updates, monitoring, report generation

## GitHub Actions Integration

Claude Code in CI/CD pipelines:

- Runs on GitHub's infrastructure (doesn't need your machine)
- Scoped to GitHub tasks: PR reviews, code generation, issue triage
- Not a general-purpose background agent — limited to GitHub context

## Channels

A newer architectural layer that enables building custom Claude Code experiences:

- Open, composable communication layer
- Enables connecting Claude Code to external messaging systems (Telegram, Slack)
- Combined with MCP and Computer Use, creates a fully extensible agent platform

## The Progression

The evolution of Claude Code's execution model:

1. **Interactive** — user prompts, Claude responds (original)
2. **[[Subagents]]** — delegate to specialized agents within a session
3. **Background agents** — parallel execution without blocking
4. **Agent loops** — autonomous cycles with verification
5. **Scheduled agents** — cron-based autonomous execution
6. **Channels** — external system integration

## See also

- [[Claude Code]]
- [[Subagents]]
- [[Hooks and Automation]]
- [[Verification-Driven Development]]
