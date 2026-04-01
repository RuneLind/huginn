---
title: GraphRAG-inspirerte forbedringer for Huginn
status: in-progress
created: 2026-04-01
---

# GraphRAG-inspirerte forbedringer

Inspirert av Microsoft GraphRAG, tilpasset Huginn sin lokale arkitektur.

## 1. Community detection (Louvain) — FERDIG

Implementert i `feature/community-detection` branch (huginn + muninn).

- Louvain community detection på similarity-matrisen via networkx
- Community-navn generert fra top tags
- Frontend: Category/Community toggle, cluster-kraft, konvekse hull-bakgrunner
- Commits: huginn 849cd41, 8989d22 | muninn 8e0f8a4, 663a31b, c41183b

## 2. LLM-basert entitetsekstraksjon (valgfri berikelse)

Bruk lokal LLM (Ollama) til å ekstrahere entiteter og relasjoner fra dokumenttekst.
Fanger implisitte relasjoner som regex ikke ser.

### Konsept

- Kjør Ollama (f.eks. qwen3:8b, qwen3.5:32b) over dokumenter i en collection
- Ekstraher entiteter (personer, konsepter, teknologier, organisasjoner)
- Ekstraher relasjoner mellom entiteter
- Lagre som JSON i samme format som eksisterende knowledge graph
- Kan merges inn i KnowledgeGraph ved oppstart

### Implementasjon

- Nytt script: `scripts/knowledge_graph/extract_entities_llm.py`
- Input: collection-mappe med dokumenter (markdown/JSON)
- Output: graph JSON (nodes + edges) kompatibelt med KnowledgeGraph
- Prompt-design: strukturert output (JSON) med entitetstyper og relasjoner
- Batch-prosessering med progress bar
- Inkrementell: hopp over allerede prosesserte dokumenter

### Verdi

- Oppdager relasjoner på tvers av dokumenter (f.eks. "denne videoen og den artikkelen handler begge om RAG")
- Bygger et semantisk nettverk utover det tags og kategorier gir
- Beriker søkeresultater med graf-kontekst

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
