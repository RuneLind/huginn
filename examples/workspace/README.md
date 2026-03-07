# Advanced: Private Domain Collections

> **Most users don't need this.** If you just want to index and search documents, use huginn directly — see the [main README](../../README.md). This pattern is for when you have multiple private domains (e.g., work + personal) with separate configs, taxonomies, and knowledge graphs.

## The pattern

Create gitignored folders inside huginn, each with its own `.git` repo:

```
huginn/                          # public repo
├── main/                        # open-source core
├── my-work/                     # gitignored, own git repo → private remote
│   ├── start.sh                 # start with work collections
│   ├── graphs/                  # domain knowledge graphs
│   ├── taxonomies/              # domain tag taxonomies
│   └── scripts/                 # daily update scripts, custom fetchers
├── my-personal/                 # gitignored, own git repo → private remote
│   └── start.sh                 # start with personal collections
├── data/                        # gitignored — all indexed data
├── start.sh                     # gitignored — your combined start script
└── .gitignore                   # includes: my-work/ my-personal/ start.sh
```

## Setup

```bash
cd huginn

# Create a private domain folder
mkdir my-work && cd my-work && git init && cd ..

# Add to .gitignore
echo "my-work/" >> .gitignore

# Create a start script
cat > my-work/start.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
KNOWLEDGE_GRAPH_PATH=my-work/graphs/my_graph.json \
uv run knowledge_api_server.py \
  --collections my-confluence my-jira \
  --port 8321
EOF
chmod +x my-work/start.sh
```

## Why this works

- **One IDE project** — open huginn, see everything
- **Clean public repo** — private folders are gitignored, never leak
- **Independent commits** — each private folder has its own `.git`, push to private remotes
- **No `--data-path` needed** — data lives in huginn's `data/` directory as usual
- **Shared venv** — all scripts use huginn's `uv run` directly

## Example start script (all collections)

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

KNOWLEDGE_GRAPH_PATH=my-work/graphs/jira_graph.json \
uv run knowledge_api_server.py \
  --collections \
    my-confluence \
    my-notion \
    my-jira \
  --port 8321
```

## MCP config

```json
{
  "mcpServers": {
    "huginn": {
      "command": "uv",
      "args": ["--directory", "/path/to/huginn", "run", "knowledge_api_mcp_adapter.py"],
      "env": {
        "KNOWLEDGE_API_URL": "http://localhost:8321",
        "KNOWLEDGE_DESCRIPTION": "Search my team docs and project knowledge"
      }
    }
  }
}
```
