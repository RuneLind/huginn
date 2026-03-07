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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from claude_cli import FRONTMATTER_RE, call_claude, extract_frontmatter, extract_json_array, get_content_excerpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TAGS_LINE_RE = re.compile(r'^tags:\s*(.+)$', re.MULTILINE)

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
    return {"categories": data["tags"], "flat": all_tags}


def has_tags(content: str) -> bool:
    """Check if frontmatter already contains a tags field."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return False
    return bool(TAGS_LINE_RE.search(match.group(1)))


def inject_tags(content: str, tags: list[str]) -> str:
    """Add or replace tags field in YAML frontmatter."""
    tags_str = ", ".join(tags)
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
    """Build the full prompt for Claude CLI."""
    taxonomy_text = json.dumps(taxonomy["categories"], indent=2, ensure_ascii=False)
    return f"""{SYSTEM_PROMPT}

Taxonomy (pick from these tags only):
{taxonomy_text}

Document:
- Title: {title}
- Breadcrumb: {breadcrumb}
- Content excerpt:
{content_excerpt}"""


def tag_document(title: str, breadcrumb: str, content_excerpt: str,
                 taxonomy: dict, model: str, timeout: int,
                 rel_path: str = "") -> list[str]:
    """Call Claude CLI to generate tags for a document."""
    prompt = build_prompt(title, breadcrumb, content_excerpt, taxonomy)
    raw_text = call_claude(prompt, model=model, timeout=timeout)

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
                     model: str, timeout: int, force: bool) -> dict:
    """Tag a single file. Returns result dict for aggregation."""
    rel_path = str(md_file.relative_to(source_dir))
    content = md_file.read_text(encoding='utf-8')

    if has_tags(content) and not force:
        return {"path": rel_path, "status": "skipped", "tags": []}

    fields = extract_frontmatter(content)
    title = fields.get('title', md_file.stem)
    breadcrumb = fields.get('breadcrumb', str(md_file.relative_to(source_dir).parent))
    excerpt = get_content_excerpt(content)

    if not excerpt.strip():
        return {"path": rel_path, "status": "skipped", "tags": []}

    try:
        tags = tag_document(title, breadcrumb, excerpt, taxonomy, model, timeout, rel_path=rel_path)
    except Exception as e:
        logger.error(f"Error tagging {rel_path}: {e}")
        return {"path": rel_path, "status": "error", "tags": []}

    return {"path": rel_path, "status": "tagged", "tags": tags, "file": md_file, "content": content}


def process_files(args):
    source_dir = Path(args.source)
    taxonomy = load_taxonomy(args.taxonomy)

    # Collect markdown files
    md_files = sorted(source_dir.rglob("*.md"))
    md_files = [f for f in md_files if ".excluded" not in f.parts]

    if args.pattern:
        md_files = [f for f in md_files if fnmatch.fnmatch(str(f.relative_to(source_dir)), args.pattern)]

    logger.info(f"Found {len(md_files)} markdown files in {source_dir}")

    if args.limit:
        md_files = md_files[:args.limit]
        logger.info(f"Limited to {args.limit} files")

    tagged_count = 0
    skipped_count = 0
    error_count = 0
    completed = 0
    tag_distribution: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_tag_single_file, f, source_dir, taxonomy,
                            args.model, args.timeout, args.force): f
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
                new_content = inject_tags(result["content"], tags)
                result["file"].write_text(new_content, encoding='utf-8')

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


def main():
    parser = argparse.ArgumentParser(description="Tag markdown documents using Claude CLI (Max subscription)")
    parser.add_argument("--source", required=True, help="Source directory with markdown files")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Print tags without writing to files")
    parser.add_argument("--force", action="store_true", help="Re-tag files that already have tags")
    parser.add_argument("--limit", type=int, help="Max number of files to process")
    parser.add_argument("--pattern", help="Glob pattern to filter files (relative to source dir)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers (default: 10)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model to use")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per document in seconds")
    args = parser.parse_args()

    # Verify claude CLI is available
    try:
        subprocess.run(["claude", "--version"], capture_output=True, timeout=5)
    except FileNotFoundError:
        print("Error: `claude` CLI not found. Install it first.", file=sys.stderr)
        sys.exit(1)

    process_files(args)


if __name__ == "__main__":
    main()
