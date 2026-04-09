---
type: concept
title: "Skills System"
aliases: ["Skills System"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, skills, automation]
sources: ["[[Sources — Claude Code]]"]
---

# Skills System

Skills are reusable, versioned collections of files containing procedural knowledge. They represent a paradigm shift from traditional tool-based approaches — encoding **how** to do things, not just **what** tools are available.

## Skills vs. Tools

| Aspect | Skills | Tools/MCPs |
|--------|--------|------------|
| Scope | Full workflow execution | Narrow, predefined actions |
| Flexibility | Runtime adaptation | Fixed schemas |
| Learning | Compound over time | Start from zero each session |
| Sharing | Portable (Git, folders) | Server-dependent |
| Context cost | Metadata only until triggered | Full schema loaded upfront |

## Two Categories

1. **Capability Uplift Skills** — fill gaps in the model's knowledge (e.g., PDF handling, PowerPoint generation, Swift concurrency patterns)
2. **Workflow Skills** — encode multi-step processes triggered by slash commands (e.g., `/commit`, `/review-pr`, `/deploy`)

## How They Work

Skills use **progressive disclosure** (see [[Context Management]]):

1. Only **metadata** (name, description, triggers) is visible to the model at runtime
2. When triggered, the agent reads `skill.md` for core instructions
3. Supporting reference files in subfolders are loaded only as needed
4. This enables **hundreds of skills** in context simultaneously without overwhelming the context window

## Creating Skills

### Structure

```
.claude/skills/my-skill/
├── skill.md          # Core instructions (triggers, rules, steps)
├── examples/         # Reference examples loaded on-demand
└── templates/        # Templates for output generation
```

### Methods

- **Claude-assisted**: ask Claude to create a skill based on a workflow you describe
- **Skill Creator skill**: Anthropic's official meta-skill for systematic creation, testing, and benchmarking
- **Manual**: write `skill.md` directly with triggers, instructions, and reference files

### Locations

- **Project-level**: `.claude/skills/` — shared with team via Git
- **User-level**: `~/.claude/skills/` — personal, available across all projects

## Self-Improving Skills

A powerful pattern where skills evolve through use:

1. Store learned corrections in markdown files alongside the skill
2. Use a "reflex skill" that analyzes completed sessions
3. Auto-extract corrections via `/reflect` command or stop [[Hooks and Automation|hooks]]
4. Track evolution with Git version control

This creates a feedback loop: mistakes made once are never repeated.

## Key Principles

- **Auto-activation**: use trigger words in the description so skills activate without explicit invocation
- **Single responsibility**: one skill, one workflow
- **Lock the skills directory**: prevent the agent from modifying its own skills during execution (unless using self-improvement pattern deliberately)
- **Commit to Git**: skills are code — version control them
- **Test with evals**: use the Skill Creator skill to benchmark performance and measure improvement

## Notable Skill Collections

- [[Anthropic]]'s official skills on GitHub
- [[GStack]] — nine workflow skills from [[Gary Tan]]
- Hookify plugin — bundles skills with hooks and MCP configs
- Community skills via the Claude Code plugin marketplace

## See also

- [[Context Management]]
- [[Hooks and Automation]]
- [[Subagents]]
- [[Claude Code]]
- [[Boris Cherny]]
