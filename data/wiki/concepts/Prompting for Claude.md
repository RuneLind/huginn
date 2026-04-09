---
type: concept
title: "Prompting for Claude"
aliases: ["Prompting for Claude"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, prompting, techniques]
sources: ["[[Sources — Claude Code]]"]
---

# Prompting for Claude

Best practices for communicating effectively with [[Claude Code]] and Claude models generally. These practices are derived from [[Anthropic]]'s official guidance and community experience.

## Golden Rules

1. **Be explicit and direct** — Claude models respond best to clear, unambiguous instructions
2. **Provide context** — explain *why*, not just *what*. Claude makes better decisions with motivation
3. **Use XML tags for structure** — Claude was extensively trained on XML-structured prompts
4. **Tell what TO do**, not what NOT to do — positive framing produces better results

## XML Tag Structure

Claude responds particularly well to XML-structured prompts:

```xml
<task>Design a microservices architecture</task>
<requirements>
- Handle 10,000+ concurrent users
- Support multiple regions
</requirements>
<constraints>
- Must use existing PostgreSQL database
- Budget under $5k/month
</constraints>
```

## Thinking Budget

Control how much reasoning Claude applies:

| Level | Keyword | Use Case |
|-------|---------|----------|
| Basic | "think" | Simple tasks |
| Moderate | "think hard" | Multi-step problems |
| Deep | "think harder" | Complex architecture |
| Maximum | "ultrathink" | Encryption, real-time systems, novel algorithms |

Higher thinking budgets cost more tokens but produce fewer errors on complex tasks. [[Boris Cherny]] recommends using Opus with thinking enabled for important work.

## Parallel Tool Use

> "For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially."

Explicitly telling Claude to parallelize tool calls can dramatically speed up tasks that involve multiple file reads, searches, or independent operations.

## Output Formatting

- Use positive framing: "Write in prose paragraphs" not "Don't use markdown"
- Specify format with XML indicators
- Prefill responses to skip preambles (in API use)

## Plan Mode

Use plan mode (`/plan`) before non-trivial tasks:

- Claude creates a detailed plan before executing
- You review and approve before any code is written
- The "interview trick": use Claude's AskUserQuestion tool to transform a rough spec into a detailed one through systematic questioning

## The Interview Trick

A powerful workflow for spec-driven development:

1. Write a rough 100-line spec
2. Tell Claude to interview you about it
3. Claude asks ~56 systematic questions about edge cases, architecture, priorities
4. The spec expands to 450+ lines of precise requirements
5. Execute the detailed spec phase by phase

## See also

- [[Context Management]]
- [[CLAUDE.md Configuration]]
- [[Claude Code]]
- [[Verification-Driven Development]]
