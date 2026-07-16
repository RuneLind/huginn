#!/usr/bin/env python3
"""
Tag markdown documents using Claude Haiku via the Claude CLI (Max subscription).

Reads markdown files from a source directory, sends title + breadcrumb + content
excerpt to Claude Haiku via `claude -p`, and writes selected tags back into YAML frontmatter.

Usage:
    # Dry-run on 10 files
    uv run scripts/tagging/tag_documents.py --source data/sources/my-confluence \
        --taxonomy scripts/tagging/my_taxonomy.json --dry-run --limit 10

    # Tag all files (10 parallel workers)
    uv run scripts/tagging/tag_documents.py --source data/sources/my-confluence \
        --taxonomy scripts/tagging/my_taxonomy.json

    # Re-tag files that already have tags
    uv run scripts/tagging/tag_documents.py --source data/sources/my-confluence \
        --taxonomy scripts/tagging/my_taxonomy.json --force

Requires `claude` CLI to be installed and authenticated (Max subscription).
"""
import argparse
import fnmatch
import json
import logging
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Repo root on path so `main.*` imports resolve regardless of cwd / invocation
# (scripts/tagging/<file>.py → parents[2] is the huginn repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from claude_cli import FRONTMATTER_RE, extract_json_array, get_content_excerpt
from main.utils.claude_cli import call_claude
from main.utils.frontmatter import read_frontmatter
from main.utils.ollama_cli import DEFAULT_MODEL as DEFAULT_OLLAMA_MODEL
from main.utils.ollama_cli import call_ollama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TAGS_LINE_RE = re.compile(r'^tags:\s*(.+)$', re.MULTILINE)

# --- HTML mode (opt-in behind --include-html) -----------------------------
# HTML explainers carry tags as a `<meta name="keywords" content="a, b, c">`
# in the head instead of YAML frontmatter. Whole-tag matcher (attr order is
# tolerated); used for both idempotent-skip detection and in-place replace.
KEYWORDS_META_RE = re.compile(
    r'<meta\b[^>]*\bname=["\']keywords["\'][^>]*>', re.IGNORECASE)
_META_CONTENT_RE = re.compile(r'\bcontent=["\'](.*?)["\']', re.IGNORECASE | re.DOTALL)
_TITLE_RE = re.compile(r'<title\b[^>]*>(.*?)</title>', re.IGNORECASE | re.DOTALL)

# Anchor fallback chain for inserting a fresh keywords meta, most-specific first.
# Each captures the anchor line's leading indentation so the inserted meta lines
# up with its neighbours. If none match, the meta is prepended at the very top
# (downstream consumers regex-match a raw prefix and don't require <head>).
_HTML_ANCHORS = [
    re.compile(r'(?P<indent>[ \t]*)<meta\b[^>]*\bname=["\']viewport["\'][^>]*>', re.IGNORECASE),
    re.compile(r'(?P<indent>[ \t]*)<meta\b[^>]*\bcharset[^>]*>', re.IGNORECASE),
    re.compile(r'(?P<indent>[ \t]*)<head\b[^>]*>', re.IGNORECASE),
]

# Body-text extraction for the prompt excerpt: drop script/style, strip tags,
# unescape entities, collapse whitespace. A bounded regex strip (no heavy
# unstructured import); same 2000-char cap as the markdown path.
_SCRIPT_STYLE_RE = re.compile(r'<(script|style)\b[^>]*>.*?</\1>', re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r'<[^>]+>')
_WS_RE = re.compile(r'\s+')


def has_html_tags(content: str) -> bool:
    """True if a non-empty `<meta name="keywords">` is already present."""
    m = KEYWORDS_META_RE.search(content)
    if not m:
        return False
    cm = _META_CONTENT_RE.search(m.group(0))
    return bool(cm and cm.group(1).strip())


def inject_html_tags(content: str, tags: list[str]) -> str:
    """Insert or replace `<meta name="keywords">`, preserving the rest of the
    file byte-for-byte.

    Replace-in-place when a keywords meta already exists (regardless of
    position); otherwise insert after the anchor chain (viewport → charset →
    <head> → prepend at top).
    """
    meta = f'<meta name="keywords" content="{", ".join(tags)}">'

    existing = KEYWORDS_META_RE.search(content)
    if existing:
        return content[:existing.start()] + meta + content[existing.end():]

    for anchor in _HTML_ANCHORS:
        m = anchor.search(content)
        if m:
            indent = m.group("indent")
            end = m.end()
            return f"{content[:end]}\n{indent}{meta}{content[end:]}"

    # No <head>/charset/viewport (headless fragment) — prepend at the top.
    return f"{meta}\n{content}"


def get_html_excerpt(content: str, max_chars: int = 2000) -> str:
    """Body text of an HTML file for the tagging prompt (2000-char cap)."""
    text = _SCRIPT_STYLE_RE.sub(" ", content)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def html_title(content: str, fallback: str) -> str:
    """The `<title>` text if present and non-empty, else the fallback."""
    m = _TITLE_RE.search(content)
    if m:
        title = unescape(m.group(1)).strip()
        if title:
            return title
    return fallback

SYSTEM_PROMPT = """\
You are a document tagger. Given a document's title, breadcrumb path, and content
excerpt, select 1-5 tags from the provided taxonomy that best describe the
document's topics.

Rules:
- Only use tags from the provided list
- Pick the most specific tags that apply
- If the document is about multiple topics, include all relevant tags (up to 5)
- If no tags fit at all, return an empty array
- Return ONLY a JSON array of tag strings, e.g. ["salg", "kundedatabase"]
- No explanation, no markdown, just the JSON array"""


def load_taxonomy(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_tags = []
    for category, tags in data["tags"].items():
        all_tags.extend(tags)
    # Optional top-level "prompt_note" is appended verbatim to the tagging prompt
    # (e.g. mimir's "first tag = owning project" rule).
    return {"categories": data["tags"], "flat": all_tags, "note": data.get("prompt_note", "")}


def has_tags(content: str) -> bool:
    """Check if frontmatter already contains a tags field."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return False
    return bool(TAGS_LINE_RE.search(match.group(1)))


def inject_tags(content: str, tags: list[str]) -> str:
    """Add or replace tags field in YAML frontmatter.

    Emits the canonical **bracketed inline** form ``tags: [a, b, c]`` — muninn's
    /wiki reader only splits bracketed arrays into per-tag chips, and huginn's
    doc-metadata consumers normalize both forms via ``parse_tags``.
    """
    tags_str = "[" + ", ".join(tags) + "]"
    match = FRONTMATTER_RE.match(content)
    if not match:
        return f"---\ntags: {tags_str}\n---\n{content}"

    fm_block = match.group(1)
    fm_end = match.end()

    if TAGS_LINE_RE.search(fm_block):
        new_fm = TAGS_LINE_RE.sub(f"tags: {tags_str}", fm_block)
    else:
        new_fm = fm_block.rstrip() + f"\ntags: {tags_str}"

    return f"---\n{new_fm}\n---{content[fm_end:]}"


def build_prompt(title: str, breadcrumb: str, content_excerpt: str, taxonomy: dict) -> str:
    """Build the full tagging prompt (backend-agnostic)."""
    taxonomy_text = json.dumps(taxonomy["categories"], indent=2, ensure_ascii=False)
    note = f"\n{taxonomy['note']}\n" if taxonomy.get("note") else ""
    return f"""{SYSTEM_PROMPT}
{note}
Taxonomy (pick from these tags only):
{taxonomy_text}

Document:
- Title: {title}
- Breadcrumb: {breadcrumb}
- Content excerpt:
{content_excerpt}"""


def call_backend(prompt: str, backend: str, model: str, ollama_model: str, timeout: int) -> str:
    """Dispatch a tagging prompt to the selected backend, returning raw model text."""
    if backend == "ollama":
        return call_ollama(prompt, model=ollama_model, timeout=timeout)
    return call_claude(prompt, model=model, timeout=timeout)


def tag_document(title: str, breadcrumb: str, content_excerpt: str,
                 taxonomy: dict, model: str, timeout: int,
                 rel_path: str = "", backend: str = "claude-cli",
                 ollama_model: str = DEFAULT_OLLAMA_MODEL) -> list[str]:
    """Call the selected backend to generate tags for a document."""
    prompt = build_prompt(title, breadcrumb, content_excerpt, taxonomy)
    raw_text = call_backend(prompt, backend, model, ollama_model, timeout)

    tags = extract_json_array(raw_text)
    if tags is None:
        logger.warning(f"Could not extract tags for {rel_path}: {raw_text[:100]}")
        return []

    valid_tags = [t for t in tags if isinstance(t, str) and t in taxonomy["flat"]]
    if len(valid_tags) != len(tags):
        invalid = set(str(t) for t in tags) - set(valid_tags)
        if invalid:
            logger.warning(f"Removed invalid tags for {rel_path}: {invalid}")
    return valid_tags


def _tag_single_file(md_file: Path, source_dir: Path, taxonomy: dict,
                     model: str, timeout: int, force: bool,
                     backend: str = "claude-cli",
                     ollama_model: str = DEFAULT_OLLAMA_MODEL) -> dict:
    """Tag a single file (markdown or HTML). Returns result dict for aggregation."""
    rel_path = str(md_file.relative_to(source_dir))
    content = md_file.read_text(encoding='utf-8')
    is_html = md_file.suffix.lower() == ".html"

    if is_html:
        if has_html_tags(content) and not force:
            return {"path": rel_path, "status": "skipped", "tags": []}
        title = html_title(content, md_file.stem)
        breadcrumb = str(md_file.relative_to(source_dir).parent)
        excerpt = get_html_excerpt(content)
    else:
        if has_tags(content) and not force:
            return {"path": rel_path, "status": "skipped", "tags": []}
        fields = read_frontmatter(content)
        title = fields.get('title', md_file.stem)
        breadcrumb = fields.get('breadcrumb', str(md_file.relative_to(source_dir).parent))
        excerpt = get_content_excerpt(content)

    if not excerpt.strip():
        return {"path": rel_path, "status": "skipped", "tags": []}

    try:
        tags = tag_document(title, breadcrumb, excerpt, taxonomy, model, timeout,
                            rel_path=rel_path, backend=backend, ollama_model=ollama_model)
    except Exception as e:
        logger.error(f"Error tagging {rel_path}: {e}")
        return {"path": rel_path, "status": "error", "tags": []}

    return {"path": rel_path, "status": "tagged", "tags": tags, "file": md_file,
            "content": content, "is_html": is_html}


def process_files(args):
    source_dir = Path(args.source)
    taxonomy = load_taxonomy(args.taxonomy)

    # Collect candidate files. Markdown always; HTML explainers only when
    # --include-html is set (opt-in — an unconditional html sweep would bulk-tag
    # explainers unsupervised on the nightly job). Hidden directories/files
    # (.claude/, .git/, …) are never candidates — mirrors the wiki reader's
    # dot-exclusion, so the tagger can't touch files no consumer will ever see.
    md_files = sorted(source_dir.rglob("*.md"))
    if args.include_html:
        md_files = sorted(md_files + list(source_dir.rglob("*.html")))
    md_files = [f for f in md_files if ".excluded" not in f.parts]
    md_files = [
        f for f in md_files
        if not any(part.startswith(".") for part in f.relative_to(source_dir).parts)
    ]

    if args.pattern:
        md_files = [f for f in md_files if fnmatch.fnmatch(str(f.relative_to(source_dir)), args.pattern)]

    for pattern in args.exclude or []:
        md_files = [f for f in md_files if not fnmatch.fnmatch(str(f.relative_to(source_dir)), pattern)]

    kind = "markdown + html" if args.include_html else "markdown"
    logger.info(f"Found {len(md_files)} {kind} files in {source_dir}")

    if args.limit:
        md_files = md_files[:args.limit]
        logger.info(f"Limited to {args.limit} files")

    tagged_count = 0
    skipped_count = 0
    error_count = 0
    # success = a file that got a usable model response (tagged, incl. empty tags).
    # Distinct from skipped (already-tagged / no excerpt) and error (call failed).
    success_count = 0
    completed = 0
    tag_distribution: dict[str, int] = {}
    written_files: list[Path] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_tag_single_file, f, source_dir, taxonomy,
                            args.model, args.timeout, args.force,
                            args.backend, args.ollama_model): f
            for f in md_files
        }

        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"Exception: {e}")
                error_count += 1
                continue

            if result["status"] == "skipped":
                skipped_count += 1
                continue
            if result["status"] == "error":
                error_count += 1
                continue

            # Reached the model and got a response (tags may still be empty).
            success_count += 1
            tags = result["tags"]
            for tag in tags:
                tag_distribution[tag] = tag_distribution.get(tag, 0) + 1

            if not tags:
                skipped_count += 1
                continue

            tagged_count += 1
            if args.dry_run:
                print(f"  {result['path']}: {tags}")
            else:
                inject = inject_html_tags if result.get("is_html") else inject_tags
                new_content = inject(result["content"], tags)
                result["file"].write_text(new_content, encoding='utf-8')
                written_files.append(result["file"].resolve())

            if completed % 20 == 0:
                logger.info(f"Progress: {completed}/{len(md_files)} processed, {tagged_count} tagged")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total files:   {len(md_files)}")
    print(f"Tagged:        {tagged_count}")
    print(f"Skipped:       {skipped_count}")
    print(f"Errors:        {error_count}")
    if tag_distribution:
        print(f"\nTag distribution:")
        for tag, count in sorted(tag_distribution.items(), key=lambda x: -x[1]):
            print(f"  {tag}: {count}")

    # Emit the exact list of files written this run (absolute paths, one per line),
    # so a caller can commit precisely those and nothing else. Non-dry-run only —
    # a dry run writes no files, so it produces no changed-files manifest.
    if args.changed_files_out and not args.dry_run:
        Path(args.changed_files_out).write_text(
            "".join(f"{p}\n" for p in written_files), encoding="utf-8")

    # Signal total failure: candidate files were attempted but every one errored
    # (e.g. the backend model is missing). Partial failures stay a success exit.
    all_failed = error_count > 0 and success_count == 0
    return {"tagged": tagged_count, "skipped": skipped_count,
            "errors": error_count, "success": success_count,
            "all_failed": all_failed}


def main():
    parser = argparse.ArgumentParser(description="Tag markdown documents using Claude CLI (Max subscription)")
    parser.add_argument("--source", required=True, help="Source directory with markdown files")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Print tags without writing to files")
    parser.add_argument("--force", action="store_true", help="Re-tag files that already have tags")
    parser.add_argument("--limit", type=int, help="Max number of files to process")
    parser.add_argument("--pattern", help="Glob pattern to filter files (relative to source dir)")
    parser.add_argument("--include-html", action="store_true",
                        help="Also tag .html explainer files via a <meta name=\"keywords\"> "
                             "tag in the head (opt-in; without it only *.md is touched).")
    parser.add_argument("--exclude", action="append", default=None, metavar="GLOB",
                        help="Glob (relative to source dir) to exclude; repeatable. "
                             "Hidden directories (.claude/, .git/, …) are always excluded.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: 10 for claude-cli, 1 for ollama — a single thread saturates the GPU)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model to use (claude-cli backend)")
    parser.add_argument("--backend", choices=["claude-cli", "ollama"], default="claude-cli",
                        help="Tagging backend (default: claude-cli)")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL,
                        help=f"Ollama model to use (ollama backend; default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per document in seconds")
    parser.add_argument("--changed-files-out", default=None,
                        help="Write the absolute path of each file actually tagged this run "
                             "(one per line) to this path. Non-dry-run only; empty when nothing "
                             "was written. Lets a caller commit exactly those files.")
    args = parser.parse_args()

    # Ollama saturates the GPU with a single thread; claude-cli parallelizes to 10.
    if args.workers is None:
        args.workers = 1 if args.backend == "ollama" else 10

    if args.backend == "claude-cli":
        # Verify claude CLI is available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, timeout=5)
        except FileNotFoundError:
            print("Error: `claude` CLI not found. Install it first.", file=sys.stderr)
            sys.exit(1)

    result = process_files(args)
    if result["all_failed"]:
        logger.error("All %d candidate file(s) failed to tag — exiting non-zero",
                     result["errors"])
        sys.exit(1)


if __name__ == "__main__":
    main()
