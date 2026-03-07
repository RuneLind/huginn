# RAG 2.0: Agentic RAG + Knowledge Graphs

**Source:** [YouTube - Introducing RAG 2.0: Agentic RAG + Knowledge Graphs (FREE Template)](https://youtu.be/p0FERNkpyHE)

## Overview

Combining two powerful retrieval strategies — **agentic RAG** and **knowledge graphs** — to create flexible knowledge retrieval systems for AI agents. The agent can reason about *how* it explores the knowledge base, choosing between vector search, graph search, or both depending on the question.

## The Problem with Naive RAG

Traditional (vanilla/naive) RAG follows a rigid pipeline:

1. Documents are chunked and embedded into a vector database
2. User query is embedded and matched against stored chunks
3. Retrieved context is force-fed to the LLM

This is **inflexible** — the agent cannot:
- Refine its search
- Do deeper dives
- Choose between multiple knowledge sources
- Reason about *how* to explore the knowledge

## Agentic RAG

Agentic RAG gives the agent the ability to **reason about how it explores the knowledge base** instead of always force-feeding context as a pre-processing step.

Key capabilities:
- The agent formulates its own queries
- It can explore different data sources (vector DBs, knowledge graphs, web search)
- It decides which tool to use based on the question type
- It can combine multiple search strategies in a single answer

Reference: [Weaviate article on Agentic RAG](https://weaviate.io/blog/agentic-rag)

## When to Use Each Strategy

| Question Type | Strategy | Example |
|---|---|---|
| Lookup on a single entity | Vector search | "What are the AI initiatives for Google?" |
| Relationship between entities | Knowledge graph | "How are OpenAI and Microsoft related?" |
| Combined entity + relationship | Both | "What are Microsoft's initiatives? How does that relate to Anthropic?" |

## Tech Stack

- **AI Agent Framework:** Pydantic AI
- **Knowledge Graph Library:** Graffiti (with Neo4j as the graph engine)
- **Vector Database:** PostgreSQL with pgvector extension (hosted on Neon)
- **API:** FastAPI
- **LLM:** OpenAI-compatible providers (OpenAI, Ollama, Gemini, Open Router)
- **AI Coding Assistant:** Claude Code

## Architecture

### Vector Database (PostgreSQL + pgvector)

- Documents chunked, embedded, and stored with pgvector
- Fast insertion (seconds)
- Good for direct lookups on individual topics

### Knowledge Graph (Neo4j + Graffiti)

- Same data represented relationally as entities and relationships
- Nodes represent companies, technologies, partnerships
- Edges represent relationships (e.g., "Amazon invested in Anthropic", "OpenAI uses Azure")
- Slower insertion (minutes) because LLMs must define entities and relationships
- Fast querying once built
- Good for exploring how things connect

### Agent

- System prompt in `agent/prompts.py` controls when to use each search strategy
- Exposed via FastAPI endpoint (`/chat/stream`)
- CLI interface for interactive use
- Tools: `vector_search`, `graph_search`

## Setup Process

1. **Prerequisites:** Python, PostgreSQL (Neon), Neo4j, LLM API key
2. **Database setup:** Run SQL schema with pgvector extension enabled
3. **Knowledge graph:** Neo4j via Docker (local-ai package) or Neo4j Desktop
4. **Environment variables:** Database URL, Neo4j credentials, LLM provider config, embedding provider config
5. **Ingest documents:** `python -m ingestion.est --clean` (processes markdown files from `documents/` folder)
6. **Run agent:** `python -m agent.api` (API server) + `python cli.py` (CLI client)

## Key Insight: Configuring Agent Behavior

The system prompt is critical — it tells the agent *when* to use each search strategy. This must be tailored to your specific data and use case. The template provides a starting point, but you need to customize the reasoning instructions based on your knowledge base.

## Building with Claude Code

The author's workflow for building this agent with Claude Code:

1. **Set up MCP servers:**
   - Crawl4AI RAG — for external library documentation (Pydantic AI, etc.)
   - Neon MCP — for automatic database management during development

2. **Plan mode** (`Shift+Tab` twice):
   - Describe what you want to build
   - Have Claude Code ask follow-up questions
   - Generate three key files:
     - `CLAUDE.md` — global rules for the coding assistant
     - `planning.md` — high-level architecture, components, tech stack, design principles
     - `task.md` — ordered list of implementation tasks

3. **Execute** (`Shift+Tab` to exit plan mode):
   - Simple prompt: "Take a look at planning and task markdown files and execute that plan"
   - Claude Code runs autonomously (35+ minutes), creating files, running SQL, writing tests
   - Approve actions as they come up (MCP calls, file writes, etc.)

4. **Examples folder:** Provide reference implementations from previous projects for Claude Code to draw inspiration from

## Cost Considerations

- Knowledge graph creation is computationally expensive (many LLM calls for entity/relationship extraction)
- A lightweight model (e.g., GPT-4.1 Nano) can be used for ingestion
- A more capable model for the actual agent reasoning
- Embedding and LLM providers can be configured independently
