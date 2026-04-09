---
type: concept
title: "Hooks and Automation"
aliases: ["Hooks and Automation"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, automation, hooks]
sources: ["[[Sources — Claude Code]]"]
---

# Hooks and Automation

Hooks are shell commands that [[Claude Code]] executes automatically at specific points in its lifecycle. They provide **guaranteed automation** that doesn't rely on Claude deciding to do something — they run deterministically, like GitHub Actions for your AI assistant.

## Hook Events

| Event | When It Fires | Common Use Cases |
|-------|--------------|------------------|
| **PreToolUse** | Before a tool runs | Security validation, blocking dangerous commands |
| **PostToolUse** | After a tool completes | Auto-formatting, linting, cleanup |
| **Stop** | Claude finishes a response | Auto-commit, metrics, verification |
| **SessionStart** | New or resumed session | Load dynamic context, check environment |
| **SubagentStop** | A [[Subagents|subagent]] completes | Result validation, quality checks |

Some sources reference up to 17 distinct hook events, including session end, pre/post compact, and notification hooks.

## Configuration

Hooks are defined in `.claude/settings.json` (project-level) or `~/.claude/settings.json` (user-level):

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": {"tool_name": "Edit|Write", "file_paths": ["*.py"]},
      "hooks": [{"type": "command", "command": "black $CLAUDE_FILE_PATHS"}]
    }]
  }
}
```

## Common Hook Patterns

### Auto-Format After Edits
Run `black`, `prettier`, or `gofmt` after every file write/edit. Catches formatting issues immediately.

### Validate Bash Commands
Block dangerous operations (`rm -rf /`, `sudo rm`, `shutdown`) before they execute. Use a Python script to parse and validate commands.

### Auto-Commit on Stop
Trigger a commit script when Claude finishes a task. Useful for [[Verification-Driven Development|verification workflows]].

### Context Preservation (Post-Compact)
When Claude compacts context (summarizes/drops older messages), critical instructions can be silently lost. A post-compact hook can re-inject essential context.

### Self-Improving [[Skills System|Skills]]
A Stop hook that triggers a "reflex" analysis after every session, extracting corrections and updating skill files automatically.

## Security Considerations

- Hooks execute with **full user permissions** — they can do anything you can do
- Always validate and sanitize inputs
- Quote shell variables: `"$VAR"` not `$VAR`
- Use absolute paths for hook scripts
- Be careful with PostToolUse hooks that trigger on Bash — avoid infinite loops

## Hookify Plugin

A community plugin that bundles common hooks into an installable package. Provides pre-built hooks for formatting, validation, and workflow automation without manual configuration.

## See also

- [[Claude Code]]
- [[CLAUDE.md Configuration]]
- [[Skills System]]
- [[Verification-Driven Development]]
- [[Subagents]]
