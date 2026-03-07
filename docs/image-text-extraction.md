# Tekstutvinning fra bilder i Confluence-sider

## Problem

Confluence-sider med diagrammer og skjermbilder lastes ned som markdown der bilder blir lenker/referanser. Bildeinnholdet indekseres ikke, så sider som kun inneholder diagrammer gir ingen søkbare treff (f.eks. "SED mottak og behandling.md").

## Dagens situasjon

- `confluence_fetcher_hierarchical.py` laster ned sider som markdown via Playwright
- Bilder i HTML blir `![alt](url)` i markdown — selve bildet lastes ikke ned
- `files_document_reader.py` bruker Unstructured som støtter OCR for lokale filer, men det hjelper ikke for innebygde Confluence-bilder

## Alternativer

### 1. Last ned bilder + OCR (Tesseract)

Utvid fetcher til å laste ned `<img>`-tags, kjøre OCR, og legge teksten inn i markdown.

- **Fordel:** Gratis, kjører lokalt, ingen API-nøkler
- **Ulempe:** Fungerer bare for tekst-i-bilder. Diagrammer gir bare boks-labels uten kontekst/flyt
- **Passer for:** Tabeller som bilder, skjermbilder med tekst

### 2. Vision LLM (Claude, GPT-4o)

Bruk en multimodal modell til å beskrive hvert bilde som tekst.

- **Fordel:** Forstår diagrammer, flytdiagrammer, arkitekturskisser — kan beskrive relasjoner og flyt
- **Ulempe:** Koster per bilde, krever API-nøkkel, tregere
- **Passer for:** Arkitekturdiagrammer, prosessflyter, UX-mockups
- **Estimert kost:** ~$0.01-0.03 per bilde (avhengig av oppløsning og modell)

### 3. Hybrid: OCR + LLM fallback

Kjør OCR først. Hvis OCR gir lite tekst (< N ord), send bildet til vision LLM.

- **Fordel:** Billig for enkle bilder, god kvalitet for diagrammer
- **Ulempe:** Mer kompleks pipeline

## Anbefalt implementering

### Steg 1: Bildednedlasting i fetcher

Utvid `confluence_fetcher_hierarchical.py`:
- Last ned bilder referert i HTML (`<img src="...">`)
- Lagre under `images/` ved siden av markdown-filen
- Erstatt bilde-URL med lokal sti i markdown

### Steg 2: Vision LLM-beskrivelse

Ny script `confluence_describe_images.py`:
- Skann markdown-filer for `![...](images/...)` referanser
- Send hvert bilde til vision LLM med prompt: "Beskriv dette diagrammet/bildet i detalj for søkeindeksering"
- Sett inn beskrivelsen under bildereferansen i markdown
- Cache resultater for å unngå re-prosessering

### Steg 3: Integrer i daglig oppdatering

Legg til bildebeskrivelses-steg i `daily_confluence_update.sh` mellom nedlasting og indeksering.

## Scope

Relevant for alle Confluence-samlinger. Mest nytteverdi for sider i:
- `Målbilde og arkitektur/` (arkitekturdiagrammer)
- `Funksjons- fagområder/` (prosessflyter, SED-diagrammer)
- `UX/` (mockups, skjermbildeflyter)
