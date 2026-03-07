# MCP Memory Optimization Guide

## Overview

This document explains how the documents-vector-search MCP server loads data, its memory footprint, and how to optimize RAM usage when running multiple collections.

---

## What Gets Loaded When Claude Desktop Starts MCP?

### Startup Process

```
Claude Desktop starts → MCP Server (Python process)
                              ↓
                    1. Load embedding model (90 MB)
                    2. Load FAISS index (~5 MB per collection)
                    3. Load document metadata (~1 MB)
                              ↓
                         Ready to search
```

### 1. Embedding Model: all-MiniLM-L6-v2

```python
# From sentence_embeder.py
self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

**What's loaded:**
- **Size:** ~90 MB (model weights)
- **Location:** Cached to `~/.cache/torch/sentence_transformers/`
- **In memory:** Yes, entire model (22M parameters)
- **First time:** Downloaded from Hugging Face
- **Subsequent starts:** Loaded from local cache (~1-2 seconds)

**Model specs:**
- Parameters: 22 million
- Dimensions: 384
- Training data: 1+ billion sentence pairs
- Languages: 50+ (including Norwegian!)

### 2. FAISS Index

```python
# From indexer_factory.py
serialized_index = persister.read_bin_file(
    f"{collection_name}/indexes/{indexer_name}/indexer"
)
self.faiss_index = faiss.deserialize_index(serialized_index)
```

**Index structure for 400 Confluence articles:**
```
400 articles
→ Split into chunks (~2000 chunks total)
→ Each chunk: 384 floats (all-MiniLM-L6-v2 dimension)
→ Total: 2000 × 384 × 4 bytes = ~3 MB

Plus:
- Document ID mapping
- Metadata pointers
```

**File size:** ~3-5 MB for 400 articles  
**In memory:** **YES - entire index!**  
**Load time:** ~0.1-0.5 seconds

#### Why the entire index must be in memory

FAISS `IndexFlatL2` performs **brute-force search** (compares with everything):

```python
# During search:
for each_embedding_in_index:
    compute_distance(query_vector, embedding)
sort_by_distance()
return_top_k()
```

This requires simultaneous access to all embeddings for speed!

### 3. Document Metadata

**What's loaded:**
- Filenames, paths, URLs
- Chunk → document mapping
- Document titles

**Size:** ~100-500 KB  
**In memory:** Partially (lazy-loading on demand)

---

## Memory Footprint

### Single MCP Server

```
┌─────────────────────────────┬──────────┐
│ Component                   │ Size     │
├─────────────────────────────┼──────────┤
│ Python process (base)       │  ~30 MB  │
│ Embedding model (MiniLM)    │  ~90 MB  │
│ FAISS index (400 docs)      │   ~5 MB  │
│ Metadata                    │   ~1 MB  │
│ Libraries (numpy, faiss)    │  ~50 MB  │
├─────────────────────────────┼──────────┤
│ TOTAL                       │ ~180 MB  │
└─────────────────────────────┴──────────┘
```

### Scaling with More Documents

| Documents | Chunks (approx) | FAISS Size | Total RAM |
|-----------|----------------|------------|-----------|
| 400       | 2,000          | 3 MB       | 180 MB    |
| 1,000     | 5,000          | 8 MB       | 185 MB    |
| 10,000    | 50,000         | 80 MB      | 250 MB    |
| 100,000   | 500,000        | 800 MB     | 950 MB    |

**Note:** Embedding model (90 MB) remains constant - independent of document count!

---

## Problem: Multiple Collections = Duplicated Model

### Original Setup (3 separate MCP servers)

```json
{
  "mcpServers": {
    "my-confluence": { ... },
    "my-jira": { ... },
    "my-vakt": { ... }
  }
}
```

**Each server loads its own copy:**

```
my-confluence  → Python process #1 → Load model (90 MB)
my-jira        → Python process #2 → Load model (90 MB)
my-vakt        → Python process #3 → Load model (90 MB)
```

**Total memory usage:**

```
┌─────────────────────────┬──────────┬──────────┐
│ Per MCP server          │ Size     │ × 3      │
├─────────────────────────┼──────────┼──────────┤
│ Python base             │   30 MB  │   90 MB  │
│ Embedding model         │   90 MB  │  270 MB  │ ← Duplicated!
│ FAISS index             │    5 MB  │   15 MB  │
│ Metadata                │    1 MB  │    3 MB  │
│ Libraries (numpy etc)   │   50 MB  │  150 MB  │
├─────────────────────────┼──────────┼──────────┤
│ TOTAL                   │  176 MB  │  528 MB  │
└─────────────────────────┴──────────┴──────────┘
```

### Why does this happen?

MCP architecture uses **separate processes** for isolation:

```
Claude Desktop
    │
    ├─── spawns Python process 1 (my-confluence)
    │       └─ Load SentenceTransformer('all-MiniLM-L6-v2')
    │
    ├─── spawns Python process 2 (my-jira)
    │       └─ Load SentenceTransformer('all-MiniLM-L6-v2')
    │
    └─── spawns Python process 3 (my-vakt)
            └─ Load SentenceTransformer('all-MiniLM-L6-v2')
```

Each process has its own memory space = no model sharing.

**Advantage:** Stability (one server crash doesn't affect others)  
**Cost:** RAM overhead

---

## Solution: Multi-Collection MCP Server

### Implementation

Created `multi_collection_search_mcp_adapter.py` that:
1. Loads embedding model **once**
2. Shares it across all collections
3. Still provides separate search tools per collection

**Key code:**

```python
# Load embedding model ONCE
shared_embedder = SentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2")

# Create indexers for all collections using the shared embedder
for collection_name in collections:
    # Load FAISS index
    serialized_index = disk_persister.read_bin_file(f"{collection_name}/indexes/...")
    
    # Create indexer with SHARED embedder (saves 90 MB per collection!)
    indexer = FaissIndexer(
        name=index_name,
        embedder=shared_embedder,  # ← Shared across all collections!
        serialized_index=serialized_index
    )
    
    searchers[collection_name] = DocumentCollectionSearcher(...)
```

### Optimized Config

```json
{
  "mcpServers": {
    "my-docs": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/huginn",
        "run",
        "multi_collection_search_mcp_adapter.py",
        "--collections",
        "my-confluence",
        "my-jira",
        "my-vakt",
        "--maxNumberOfChunks",
        "50"
      ]
    }
  }
}
```

### Memory Savings

```
┌──────────────────────────────┬─────────┬──────────┐
│ Setup                        │ RAM     │ Savings  │
├──────────────────────────────┼─────────┼──────────┤
│ 3 separate servers (old)     │ 528 MB  │ -        │
│ 1 multi-collection (new)     │ 203 MB  │ 325 MB   │
│                              │         │ (62% ↓)  │
└──────────────────────────────┴─────────┴──────────┘
```

**New breakdown:**
```
Python base:        30 MB × 1 =  30 MB
Embedding model:    90 MB × 1 =  90 MB ← Shared!
FAISS indexes:       5 MB × 3 =  15 MB
Metadata:            1 MB × 3 =   3 MB
Libraries:          50 MB × 1 =  50 MB
Overhead:                      ~15 MB
─────────────────────────────────────
TOTAL:                        203 MB
```

### Bonus: Faster Startup

**Old (sequential loading):**
```
Start process 1 → Load model (1.5s)
Start process 2 → Load model (1.5s)  
Start process 3 → Load model (1.5s)
Total: ~4.5s
```

**New (single loading):**
```
Start process 1 → Load model once (1.5s)
                  Load all indexes (0.3s)
Total: ~1.8s
```

**2.7 seconds faster startup!** ⚡

---

## Migration Instructions

### 1. Backup existing config

```bash
cp "~/Library/Application Support/Claude/claude_desktop_config.json" \
   "~/Library/Application Support/Claude/claude_desktop_config_backup.json"
```

### 2. Test the new adapter (optional)

```bash
cd /path/to/huginn

uv run multi_collection_search_mcp_adapter.py \
  --collections my-confluence my-jira my-vakt \
  --maxNumberOfChunks 50
```

### 3. Update Claude config

Replace the contents of `claude_desktop_config.json` with the optimized version in `claude_desktop_config_optimized.json`:

```bash
cp "~/Library/Application Support/Claude/claude_desktop_config_optimized.json" \
   "~/Library/Application Support/Claude/claude_desktop_config.json"
```

### 4. Restart Claude Desktop

---

## Functionality Remains Identical

Claude still gets 3 separate tools:
```
- search_my-confluence
- search_my-jira  
- search_my-vakt
```

But now everything runs in one process with a **shared embedding model**!

---

## How the Sharing Works

```python
# One embedding model instance
shared_embedder = SentenceEmbedder("all-MiniLM-L6-v2")  # 90 MB

# All indexers use the same model
confluence_indexer = FaissIndexer(embedder=shared_embedder)
jira_indexer = FaissIndexer(embedder=shared_embedder)
vakt_indexer = FaissIndexer(embedder=shared_embedder)
```

**Result:**
- Search in Confluence: `shared_embedder.encode(query)`
- Search in Jira: `shared_embedder.encode(query)` ← Same instance!
- Search in Vakt: `shared_embedder.encode(query)` ← Same instance!

---

## Search Performance

### Query execution time:

```
1. Your question: "What are cross-border workers?"
   ↓
2. Embedding model: [0.23, -0.45, ...] (0.01s)
   ↓
3. FAISS search: Compare with all 2000 embeddings (0.001s)
   ↓
4. Return top 100 chunks
   ↓
5. MCP sends to Claude Desktop
```

**Total time:** ~10-20 milliseconds

No disk access needed! Everything in RAM = lightning fast ⚡

---

## Further Optimizations for Large Collections

If you reach 100k+ documents (1 GB+ index):

### 1. Use HNSW instead of Flat

```python
# IndexFlatL2 → IndexHNSWFlat
index = faiss.IndexHNSWFlat(384, 32)  # 32 = M-parameter
```

**Advantages:**
- 10-100x faster for large indexes
- ~5% less accuracy (acceptable trade-off)
- ~20% more memory

### 2. Use smaller embedding model

```python
# Options:
# all-MiniLM-L6-v2 (384 dim, 90 MB) ← Current
# paraphrase-MiniLM-L3-v2 (384 dim, 50 MB, faster but weaker)
# all-mpnet-base-v2 (768 dim, 420 MB, better quality)
```

### 3. Split into multiple collections

```
my-confluence-2024
my-confluence-2023
my-jira-bugs
my-jira-features
```

Claude can search multiple collections in parallel!

---

## Comparison: Database vs FAISS

### PostgreSQL with pgvector

```
├─ Startup: Connect to DB (10-100ms)
├─ Search: SQL query + vector matching (50-500ms)
├─ Scaling: Good for millions of rows
└─ RAM: Caching depends on DB config
```

### FAISS in-memory

```
├─ Startup: Load entire index (100-500ms)
├─ Search: Pure Python/C++ (1-10ms)
├─ Scaling: Perfect for 1k-1M embeddings
└─ RAM: Everything in memory (but efficient!)
```

**For MCP use case:** FAISS is superior! 🏆

- Instant responses (10ms)
- No database management
- Simple deployment
- Perfect for 1k-100k documents

---

## Technical Details

### Embedding Model: all-MiniLM-L6-v2

**What is it?**
An AI model that converts text to 384-dimensional vectors that capture semantic meaning.

**Example:**
```python
model.encode("grensearbeidere") 
→ [0.23, 0.45, -0.12, ...]

model.encode("cross-border workers")
→ [0.22, 0.46, -0.13, ...]  ← Nearly identical!
```

The model learned that these mean the same thing **without** a translation database!

**Trained on:** 1+ billion sentence pairs across 50+ languages

### FAISS Index: IndexFlatL2

**What is it?**
A vector search library from Meta (Facebook) that finds similar items in large datasets.

**How it works:**
1. Each document chunk gets an embedding (384 numbers)
2. Query gets embedded the same way
3. FAISS computes distance to ALL embeddings
4. Returns top K closest matches

**L2 distance formula:**
```
distance = sqrt(sum((query[i] - doc[i])^2 for i in range(384)))
```

Smaller distance = more similar!

### Is all-MiniLM-L6-v2 an LLM?

**No!** It's an embedding model:

| Feature | LLM (GPT-4) | Embedding Model (MiniLM) |
|---------|-------------|---------------------------|
| **Size** | ~1.7T params | 22M params |
| **Input** | Text | Text |
| **Output** | New text | Vector/array of numbers |
| **Can generate?** | ✅ Yes | ❌ No |
| **Can understand similarity?** | ❌ No (must generate) | ✅ Yes (directly) |
| **Speed** | Slow (generates word-by-word) | Fast (one pass) |
| **Use case** | Conversation, analysis | Search, matching |

**They work together in RAG:**
1. Embedding model finds relevant documents
2. LLM reads and answers based on those documents

---

## Summary

### Key Points

1. **Embedding model loaded once per MCP process** (~90 MB)
2. **FAISS index fully loaded in RAM** (~5 MB per 400 docs)
3. **Multiple collections = model duplication** (528 MB for 3 collections)
4. **Solution: Multi-collection adapter** (saves 325 MB, 62% reduction)
5. **Search performance: ~10ms** (instant in Claude Desktop)
6. **Shared model maintains identical functionality**

### Files Created

- `multi_collection_search_mcp_adapter.py` - Optimized MCP server
- `claude_desktop_config_optimized.json` - New Claude config
- `MCP_MEMORY_OPTIMIZATION.md` - This document

### Commands

```bash
# Test optimized server
uv run multi_collection_search_mcp_adapter.py \
  --collections my-confluence my-jira my-vakt

# Update Claude config
cp claude_desktop_config_optimized.json claude_desktop_config.json

# Restart Claude Desktop
```

---

## References

- **FAISS:** https://github.com/facebookresearch/faiss
- **Sentence Transformers:** https://www.sbert.net/
- **MCP Protocol:** https://modelcontextprotocol.io/
- **documents-vector-search:** https://github.com/shnax0210/documents-vector-search
