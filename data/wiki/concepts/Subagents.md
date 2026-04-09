---
type: concept
title: "Subagents"
aliases: ["Subagents"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, agents, architecture]
sources: ["[[Sources — Claude Code]]"]
---

# Subagents

Subagents are specialized AI assistants within [[Claude Code]], each with its own isolated context window. They are the primary mechanism for [[Context Management|preserving context]] in the main conversation and for parallelizing work.

## Why Subagents Matter

1. **Context preservation** — subagent work doesn't pollute the main conversation's context
2. **Parallel execution** — run 5-10 agents simultaneously on independent tasks
3. **Specialized expertise** — custom prompts and tool access for specific domains
4. **Flexible permissions** — limit each agent's tool access to only what it needs

## Creating Subagents

### File Format

```markdown
---
name: code-reviewer
description: Use PROACTIVELY for code reviews
tools: Read, Grep, Glob, Bash
model: inherit
---

You are a senior code reviewer...
[detailed instructions]
```

### Locations

- **Project-level**: `.claude/agents/*.md` — shared with team via Git
- **User-level**: `~/.claude/agents/*.md` — personal, available across all projects

## Essential Subagent Types

| Agent | Purpose | Key Tools |
|-------|---------|-----------|
| Code Reviewer | Quality, security, maintainability checks | Read, Grep, Glob |
| Debugger | Root cause analysis, error fixing | Read, Bash, Grep |
| Test Runner | Run tests, fix failures | Bash, Read, Edit |
| File Finder | Locate task-related files | Glob, Grep, Read |
| Security Scanner | Vulnerability detection | Read, Grep, Bash |
| Research Agent | Explore codebases, find patterns | Read, Grep, Glob, WebSearch |

## Best Practices

- **Use "PROACTIVELY" or "MUST BE USED" in descriptions** — this triggers automatic invocation when the agent detects a matching task
- **Single, clear responsibility** per agent — don't create Swiss-army-knife agents
- **Generate with Claude first, then customize** — ask Claude to create an agent definition, then refine
- **Limit tool access** — only grant tools the agent actually needs
- **Version control** project-level agents alongside your code

## Agent Teams

Multiple subagents can work as coordinated teams:

- Each agent maintains its own token context window
- Agents share a workspace (filesystem) but not conversation context
- A main orchestrator delegates tasks to specialist agents
- Results are returned to the orchestrator for synthesis

[[Boris Cherny]] runs **5 Claude Code sessions in parallel** with numbered tabs — a human-orchestrated version of agent teams.

## Background Agents

A newer capability that enables:

- Parallel execution of multiple coding tasks
- Agents that communicate and share information
- Building interdependent components simultaneously
- Silent task execution without blocking the main session

## See also

- [[Context Management]]
- [[Skills System]]
- [[Agent Loops]]
- [[Claude Code]]
- [[Verification-Driven Development]]
