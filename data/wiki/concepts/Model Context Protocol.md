---
type: concept
title: "Model Context Protocol"
aliases: ["Model Context Protocol"]
created: 2026-04-06
updated: 2026-04-06
tags: [claude-code, mcp, tools, integration]
sources: ["[[Sources — Claude Code]]"]
---

# Model Context Protocol

MCP (Model Context Protocol) is an open standard for connecting AI models to external tools and data sources. In [[Claude Code]], MCP is deeply integrated — Claude Code is both an **MCP client** (consuming tools from servers) and an **MCP server** (exposable to other agents).

## How MCP Works

MCP servers expose tools that Claude can call. Each tool has a schema (name, description, parameters) that tells Claude what it does and how to use it. When Claude decides to use a tool, it sends a request to the MCP server and receives a response.

## Configuration

### Project-Level (`.mcp.json` at project root)
Shared team tools. API keys go in environment variables, never hardcoded.

### User-Level (`~/.claude/mcp.json`)
Personal tools available across all projects.

## The Context Problem

Every MCP tool call dumps its **full output directly into the context window**. With many tools or verbose outputs, context fills rapidly — degrading model performance (see [[Context Management]]).

### Solutions

- **Context Mode MCP Server**: intercepts tool outputs and stores them externally, returning only summaries to the context window. Can reduce token consumption by up to 95%.
- **Dynamic tool discovery**: instead of loading all tools at startup, Claude discovers and loads only 3-5 relevant tools per prompt. Triggers when context hits ~10% threshold.
- **Tool search**: MCP tools can be deferred — only their names are listed initially, and full schemas are loaded on-demand.

## Notable MCP Servers

- **Context7**: provides up-to-date library documentation, reducing hallucination about API syntax
- **Chrome/Browser integration**: extends Claude Code into browser workflows for UI validation and web testing
- **Salesforce, GA4, databases**: connect Claude to business data
- **Knowledge bases**: local vector search over documents (like qmd)

## MCP + Skills + Agents

The three extension layers of Claude Code:

1. **MCP** — connects to external systems and data
2. **[[Skills System|Skills]]** — encodes procedural knowledge and workflows
3. **[[Subagents]]** — specialized agents with isolated context

These compose: a skill can invoke MCP tools, an agent can use skills, and the main session orchestrates agents.

## Plugins

Claude Code plugins bundle skills, [[Hooks and Automation|hooks]], agent definitions, and MCP configs into a single installable unit. Install with `/plugin install <name>`. There are 42+ official plugins available.

## See also

- [[Context Management]]
- [[Skills System]]
- [[Claude Code]]
- [[Hooks and Automation]]
