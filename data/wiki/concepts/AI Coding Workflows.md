---
type: concept
title: "AI Coding Workflows"
aliases: ["AI Coding Workflows"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, methodology, workflows, bmad, gsd]
sources: ["[[Sources — Claude Code]]"]
---

# AI Coding Workflows

Structured methodologies for using AI coding agents effectively, moving beyond "vibe coding" (unstructured prompting) toward disciplined, repeatable processes.

## The Problem with Vibe Coding

Unstructured AI coding — prompting without a plan, accepting output without verification — produces:
- Inconsistent quality
- Architectural drift
- Accumulated technical debt
- Difficult-to-debug failures

The methodologies below all aim to impose structure without sacrificing speed.

## BMAD Method

**B**uild, **M**anage, **A**rchitect, **D**evelop — a multi-agent framework with specialized roles across two phases.

### Planning Phase Agents

| Agent | Role | Output |
|-------|------|--------|
| Mary | Business Analyst | Project brief |
| John | Product Manager | Product Requirements Document (PRD) |
| Winston | Architect | Tech stack, data models, coding standards, API specs |
| Sarah | Project Manager | Document sharding into epics |

### Development Phase Agents

| Agent | Role | Output |
|-------|------|--------|
| Bob | Scrum Master | Draft user stories |
| James | Developer | Code implementation with task checklists |
| Quinn | QA Engineer | Validation against PRD |
| Taylor | Tester | Acceptance testing |

**Key insight**: BMAD is modular — use the agents that fit your project size. A solo developer might only use Winston + James + Quinn.

Version 4.3 deploys 7 specialized agents. Available as an open-source framework.

## GSD Framework (Get Stuff Done)

A lighter-weight alternative to BMAD with a three-document structure:

1. **Project file** — source of truth (what and why)
2. **Roadmap** — tactical phases and deliverables
3. **State document** — progress tracking and metrics

### Execution Pattern

- Spawn fresh [[Subagents]] with 200K context for each 2-3 atomic tasks
- **Verification → validation → commit** cycle (aligns with [[Verification-Driven Development]])
- Human validation required before proceeding to next phase
- Uses XML formatting for structured communication with Claude

## Superpowers Plugin

A middle ground between BMAD's complexity and unstructured coding:

- Enforces a workflow: brainstorming → planning → implementation → review
- Installed as a Claude Code plugin
- Less overhead than BMAD, more structure than raw prompting
- One source committed to a 30-day trial replacing it with [[GStack]]

## Spec-Driven Development

The "interview trick" (see [[Prompting for Claude]]):

1. Start with a rough spec (~100 lines)
2. Let Claude interview you systematically (~56 questions)
3. Spec expands to 450+ lines of precise requirements
4. Divide into phases/tasks
5. Execute phase by phase with [[Verification-Driven Development|verification]]

## Comparative Overview

| Framework | Complexity | Best For | Key Strength |
|-----------|-----------|----------|--------------|
| BMAD | High | Large projects, teams | Comprehensive role coverage |
| GSD | Medium | Solo devs, medium projects | Clean document structure |
| Superpowers | Low-Medium | Quick projects | Easy setup, enforced flow |
| Spec-Driven | Medium | Feature development | Requirements precision |
| Vanilla Claude | Low | Small tasks, experienced users | Speed, flexibility |

Multiple sources note that by 2026, vanilla [[Claude Code]] with Opus handles **90% of daily work** — frameworks are most valuable for larger, multi-phase projects.

## See also

- [[Verification-Driven Development]]
- [[Claude Code]]
- [[Subagents]]
- [[Prompting for Claude]]
- [[Boris Cherny]]
