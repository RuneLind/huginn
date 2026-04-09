---
type: entity
title: "Boris Cherny"
aliases: ["Boris Cherny"]
created: 2026-04-06
updated: 2026-04-06
tags: [person, anthropic, claude-code-creator]
sources: ["[[Sources — Claude Code]]"]
---

# Boris Cherny

Creator of [[Claude Code]] at [[Anthropic]]. His practices and philosophy are extensively documented across multiple YouTube sources and have become the de facto standard for effective Claude Code usage.

## Background

- TypeScript expert — TypeScript's type system shaped his coding philosophy, with type signatures being the most important aspect of clean code
- This type-first mindset carries into his approach to AI-assisted development

## The One Rule He Never Breaks

**[[Verification-Driven Development]]**: always give Claude ways to verify its own work. This is his most-cited principle across all sources.

## Workflow

1. Run **5 Claude Code sessions in parallel** with numbered tabs for notification tracking
2. Use **plan mode first** to validate approach before execution
3. Switch to **auto-accept edits mode** after plan approval
4. Choose **Opus with thinking enabled** — slower but fewer errors
5. Deploy [[Subagents]] for architecture verification, refactoring, build validation
6. Linters/formatters clean the remaining ~10% of errors before CI

## Tools He Uses

- **Clawd Chrome extension** (Ralph Wiggum plugin) — for visual UI verification
- **Claude Code GitHub Actions** — automated PR reviews
- **Stop hooks** — trigger verification when Claude finishes
- **Background agents** — silent parallel task execution

## Key Advice

On [[CLAUDE.md Configuration]]:
- Keep at **~2,500 tokens** — lean is better
- Include explicit "things not to do" lists
- Update **multiple times per week** when mistakes happen
- Capture PR review feedback automatically

On quality:
> "Maintain the same quality standards for model-generated code as human-written code. Use the model to improve and clean up the code rather than lowering the bar."

On [[Skills System|skills]]:
- Two types: auto-invoked domain knowledge and workflow skills (slash commands)
- Commit skills to Git — reuse across every project
- Any repeated pattern should become a skill

## Beyond Coding

Boris notes Claude Code applications extend well beyond code:
- Data analysis
- DBT pipelines
- Salesforce integration
- Non-technical users (sales teams)

## See also

- [[Claude Code]]
- [[Anthropic]]
- [[Verification-Driven Development]]
- [[CLAUDE.md Configuration]]
- [[Skills System]]
