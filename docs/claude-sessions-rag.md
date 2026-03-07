# Claude Code Sessions RAG - Research & Design

## Goal

Index all Claude Code session transcripts into a vector search collection so we can:
1. Search past conversations semantically
2. Reconstruct context from multiple sessions on a topic
3. Power an agent that synthesizes knowledge across sessions

## Session Storage Format

### File Locations

| Path | Purpose | Format |
|------|---------|--------|
| `~/.claude/projects/<project-path>/<uuid>.jsonl` | Session transcripts | JSONL |
| `~/.claude/projects/<project-path>/sessions-index.json` | Session metadata (summary, dates, branch) | JSON |
| `~/.claude/projects/<project-path>/<uuid>/subagents/agent-<id>.jsonl` | Subagent transcripts | JSONL |
| `~/.claude/history.jsonl` | Global prompt history | JSONL |
| `~/.claude/transcripts/ses_*.jsonl` | Legacy transcripts (older format) | JSONL |

**Project directory naming:** Absolute path with `/` replaced by `-`, e.g. `/path/to/my-project` → `-path-to-my-project`

### JSONL Message Types

Each line is a JSON object. Messages link via `parentUuid`/`uuid` forming a tree.

#### `user` — User prompts and tool results
```json
{
  "type": "user",
  "uuid": "3fa856ba-...",
  "parentUuid": "3541c233-...",
  "timestamp": "2026-02-20T21:22:38.874Z",
  "sessionId": "022ab21f-...",
  "cwd": "/path/to/huginn",
  "gitBranch": "feature/confluence-cleanup-pipeline",
  "version": "2.1.49",
  "isSidechain": false,
  "isMeta": false,
  "message": {
    "role": "user",
    "content": "Your prompt text here"
  }
}
```

When carrying a tool result, `message.content` is an array with `tool_result` objects.

#### `assistant` — Claude responses, tool calls, thinking
```json
{
  "type": "assistant",
  "uuid": "47f32d1b-...",
  "parentUuid": "a4333c37-...",
  "timestamp": "2026-02-20T21:24:42.453Z",
  "message": {
    "model": "claude-opus-4-6",
    "role": "assistant",
    "content": [
      { "type": "text", "text": "The response text..." },
      { "type": "tool_use", "name": "Bash", "input": { "command": "..." } },
      { "type": "thinking", "thinking": "reasoning here..." }
    ],
    "usage": { "input_tokens": 1, "output_tokens": 10 }
  }
}
```

#### `system` — Metadata events
Subtypes: `turn_duration`, `compact_boundary`

#### `progress` — Hook/tool progress events

#### `file-history-snapshot` — File backup state

#### `queue-operation` — Background task queue

#### `pr-link` — Links session to a GitHub PR

### Sessions Index (`sessions-index.json`)
```json
{
  "version": 1,
  "entries": [
    {
      "sessionId": "ca76f0cc-...",
      "fullPath": "~/.claude/projects/.../<uuid>.jsonl",
      "firstPrompt": "First user prompt text...",
      "summary": "Auto-generated session summary",
      "messageCount": 5,
      "created": "2026-01-17T12:19:18.447Z",
      "modified": "2026-01-17T12:20:24.051Z",
      "gitBranch": "feature/branch-name",
      "projectPath": "/path/to/my-project"
    }
  ],
  "originalPath": "/path/to/my-project"
}
```

### Common Fields

| Field | Description |
|-------|-------------|
| `type` | Message type discriminator |
| `uuid` / `parentUuid` | Message tree structure |
| `timestamp` | ISO 8601 |
| `sessionId` | UUID for the session |
| `cwd` | Working directory |
| `gitBranch` | Active git branch |
| `version` | Claude Code version |
| `isSidechain` | True for subagent messages |
| `isMeta` | True for internal/meta messages |

## Design Decisions

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| Granularity | One markdown file per session | Keeps conversation context together; chunks handle precision |
| Content | User + assistant text + thinking | Thinking captures decisions and reasoning |
| Tool calls | Include tool name + brief summary, skip verbose output | Tool results are too large but tool names give context |
| Subagents | Skip (isSidechain=true) | Too verbose, low signal-to-noise |
| Meta messages | Skip (isMeta=true) | Internal plumbing, not useful for search |
| Update | Incremental based on sessions-index.json modified time | Same pattern as Notion/Confluence daily updates |

## Implementation Plan

### Phase 1: Converter Script
`scripts/claude_sessions/claude_sessions_to_markdown.py`
- Parse all `sessions-index.json` files under `~/.claude/projects/`
- For each session, read the JSONL and produce a markdown file
- Frontmatter: sessionId, project, branch, date, summary, firstPrompt
- Body: user/assistant turns as markdown sections
- Skip: tool results, sidechains, meta messages, progress, system events
- Include: tool_use name/description as brief notes
- Include: thinking blocks (collapsed or prefixed)
- Incremental: only convert sessions newer than last run

### Phase 2: Index with existing pipeline
```bash
uv run files_collection_create_cmd_adapter.py \
  --basePath ./data/sources/claude-sessions \
  --collection claude-sessions
```

### Phase 3: Daily update script
Similar to `daily_notion_update.sh` — convert new sessions, then update collection.

### Phase 4: Context reconstruction agent
An agent/skill that:
1. Takes a topic query
2. Searches the claude-sessions collection
3. Pulls relevant chunks from multiple sessions
4. Synthesizes a coherent context summary

## References

- [Claude Code hidden conversation history](https://kentgigger.com/posts/claude-code-conversation-history)
- [simonw/claude-code-transcripts](https://github.com/simonw/claude-code-transcripts) — HTML export tool
- [ZeroSumQuant/claude-conversation-extractor](https://github.com/ZeroSumQuant/claude-conversation-extractor) — Python extractor
- [Inside Claude Code session format (Medium)](https://databunny.medium.com/inside-claude-code-the-session-file-format-and-how-to-inspect-it-b9998e66d56b)
- [Building Conversation Search Skill](https://alexop.dev/posts/building-conversation-search-skill-claude-code/)
