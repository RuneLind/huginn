---
type: concept
title: "CLAUDE.md Configuration"
aliases: ["CLAUDE.md Configuration"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, configuration, memory]
sources: ["[[Sources — Claude Code]]"]
---

# CLAUDE.md Configuration

The `CLAUDE.md` file is [[Claude Code]]'s project memory — a markdown file that loads automatically at the start of every session. It tells Claude who it is in the context of your project, what conventions to follow, and what mistakes to avoid.

## Memory Hierarchy

CLAUDE.md files cascade with increasing specificity:

| Priority | Type | Location | Scope |
|----------|------|----------|-------|
| 1 (highest) | Enterprise | `/Library/Application Support/ClaudeCode/CLAUDE.md` | Organization-wide |
| 2 | Project root | `./CLAUDE.md` | Current project |
| 3 | Subdirectory | `./src/CLAUDE.md` | Module-specific |
| 4 | User | `~/.claude/CLAUDE.md` | Personal, all projects |

More specific files take precedence. You can use hierarchical CLAUDE.md files: root for global rules, subdirectory files for module-specific conventions.

## Six Core Sections

Analysis of 2,500+ repositories (cited in multiple sources) identified six essential sections:

1. **Commands** — place early in the file; common build/test/lint commands
2. **Project structure** — directory layout and key files
3. **Tech stack versions** — specific versions, not vague ("React 19", not "React")
4. **Code style** — naming conventions, patterns, preferences
5. **Git workflows** — branching strategy, commit message format, PR process
6. **Boundaries** — what Claude should NOT do

## What to Include

- Development standards specific to your project
- Architecture decisions and patterns
- Common commands and workflows
- Team preferences and conventions
- **"Things not to do" lists** — add whenever Claude makes mistakes
- File imports for longer references: `@docs/api-standards.md`

## What to Exclude

- Basic programming concepts (Claude already knows these)
- Frequently changing details (current sprint tasks)
- Secrets (API keys, passwords) — never put these in CLAUDE.md
- Information easily found in documentation

## Key Practices

- **Keep it lean** — ~2,500 tokens is ideal. Bloated CLAUDE.md files hurt more than they help
- **Update frequently** — multiple times per week when Claude makes errors. The "things not to do" list should grow from real friction
- **Start with `/init`** — generates a starter CLAUDE.md by analyzing your project
- **Use imports** — reference external files (`@docs/api-standards.md`) rather than inlining everything
- **Evolve through friction** — don't try to write the perfect CLAUDE.md upfront. Let it grow from actual problems

> [!warning]
> A recent study ("Evaluating agents.md") found convention files can **hurt** coding agents in some cases. However, multiple sources note this doesn't apply to personal assistant use cases or well-maintained CLAUDE.md files. The risk is with bloated, stale, or contradictory instructions.

## Claude CTX

An open-source tool for managing multiple Claude Code configurations — switching between different `settings.json`, `CLAUDE.md`, and MCP configurations for different clients, projects, or workflows.

## See also

- [[Context Management]]
- [[Claude Code]]
- [[Skills System]]
- [[Hooks and Automation]]
