---
title: GraphRAG-inspirerte forbedringer for Huginn
status: 2 of 3 complete
created: 2026-04-01
updated: 2026-04-03
---

# GraphRAG-inspirerte forbedringer

Inspirert av Microsoft GraphRAG, tilpasset Huginn sin lokale arkitektur.
Se `docs/graph-enhanced-rag.html` for interaktiv teknisk dokumentasjon.

## 1. Community detection (Louvain) — FERDIG

Implementert i `feature/community-detection` branch (huginn + muninn).

- Louvain community detection på similarity-matrisen via networkx
- Percentile-basert terskel (75th) for å håndtere homogene collections
- Community-navn generert fra top tags, deduplisert med representative docs
- Frontend: Category/Community toggle, cluster-kraft, konvekse hull-bakgrunner, named chips
- Commits: huginn 849cd41..eb9aa60 | muninn 8e0f8a4..f2495da

## 2. LLM-basert entitetsekstraksjon — FERDIG

Implementert og kjørt på 3 collections. Totalt 8,096 entiteter, 15,022 relasjoner.

- Script: `scripts/knowledge_graph/extract_entities_llm.py`
- Bruker Ollama (qwen3.5) lokalt — ingen data forlater maskinen
- Inkrementell cache (.cache.json) — kan stoppes og gjenopptas
- Auto-detekteres fra huginn-jarvis/ og huginn-nav/ ved server-oppstart
- Entity detection i søk via label-matching på entity:* noder
- Query expansion via 1-hop graf-traversering
- Graph context annoteres på hvert søkeresultat

### Ekstraksjonsstatus

| Collection | Docs | Entiteter | Relasjoner | Lagret i |
|---|---|---|---|---|
| youtube-summaries | 402 | 2,383 | 3,118 | huginn-jarvis |
| melosys-confluence | 296 | 1,348 | 1,888 | huginn-nav |
| jira-issues | 2,158 | 5,637 | 10,016 | huginn-nav |

Commits: huginn 8b683d1..075a1f2

## 3. Global search (map-reduce over cluster-sammendrag)

Besvarer brede spørsmål ("hva vet vi om X?") ved å aggregere cluster-sammendrag.

### Konsept

- Forutsetning: community detection + cluster-sammendrag
- For hvert community: generer et sammendrag (kan bruke Haiku som tagging-pipelinen)
- Ved "global" spørsmål: map over alle sammendrag, reduce til ett svar
- Nytt API-endepunkt: `/api/global-search?q=...`

### Implementasjon

- Generer community-sammendrag ved indeksering (eller on-demand)
- Lagre sammendrag i collection metadata
- Map-fase: send hvert sammendrag + spørsmål til LLM, få partial svar
- Reduce-fase: kombiner partial svar til ett helhetlig svar
- Cache sammendrag, oppdater ved re-indeksering

### Verdi

- Besvarer "hva vet vi om trygdeavgift?" på tvers av tusenvis av dokumenter
- Standard RAG finner enkeltchunks; global search syntetiserer på tvers
- Nyttig for onboarding, statusoversikt, temaanalyse
