---
type: concept
title: "Verification-Driven Development"
aliases: ["Verification-Driven Development"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, methodology, best-practices]
sources: ["[[Sources — Claude Code]]"]
---

# Verification-Driven Development

The one rule [[Boris Cherny]] (creator of [[Claude Code]]) never breaks: **always give Claude ways to verify its own work.**

## Core Principle

AI-generated code is probabilistic. Without verification, errors accumulate silently. The fix is not to lower your quality bar — it's to build verification into every step of the workflow.

> "Maintain the same quality standards for model-generated code as human-written code. Use the model to improve and clean up the code rather than lowering the bar."
> — [[Boris Cherny]]

## Boris's Verification Chain

1. **Plan mode first** — validate approach before writing code
2. **Test cases** — give Claude tests to run against its own output
3. **Visual checks** — use the Clawd Chrome extension (Ralph Wiggum plugin) for UI validation
4. **Linters and formatters** — catch the remaining ~10% of errors before CI
5. **Stop [[Hooks and Automation|hooks]]** — trigger automated verification when Claude finishes
6. **Claude Code GitHub Actions** — automated PR reviews using Claude

## Boris's Workflow

1. Run **5 Claude Code sessions in parallel** with numbered tabs
2. Use **plan mode** to validate approach before execution
3. Switch to **auto-accept edits mode** after plan approval
4. Choose **Opus with thinking enabled** — slower but fewer errors
5. Deploy [[Subagents]] for architecture verification, refactoring, build validation
6. Linters/formatters clean the final output

## Forms of Verification

| Method | What It Catches | Cost |
|--------|----------------|------|
| Test suites | Logic errors, regressions | Low (automated) |
| Type checking | Type errors, interface violations | Low (automated) |
| Linting | Style, common patterns | Low (automated) |
| Visual comparison | UI regressions, layout issues | Medium (needs browser) |
| Plan review | Architecture mistakes, wrong approach | Low (human review) |
| PR review (CI) | Cross-cutting concerns | Medium (CI pipeline) |

## Relationship to Other Methodologies

Verification-driven development is complementary to other [[AI Coding Workflows]]:

- **[[AI Coding Workflows|BMAD]]** has Quinn (validation agent) and Taylor (acceptance testing) as explicit verification steps
- **[[AI Coding Workflows|GSD]]** includes a verification-validation-commit cycle
- **Spec-driven development** front-loads verification by making requirements explicit

## Anti-Patterns

- Accepting Claude's output without running it
- Skipping tests because "it looks right"
- Disabling linters to avoid friction
- Not reviewing AI-generated code with the same rigor as human code

## See also

- [[Boris Cherny]]
- [[Claude Code]]
- [[AI Coding Workflows]]
- [[Hooks and Automation]]
- [[Subagents]]
