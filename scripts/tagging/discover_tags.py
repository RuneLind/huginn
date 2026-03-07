#!/usr/bin/env python3
"""
Discover natural tags from document content using Claude Haiku (free-form).

Sends a sample of documents to Haiku and asks for free-form topic tags.
Aggregates results to help build a data-driven taxonomy.

Usage:
    # Sample 100 random files, 10 in parallel
    uv run scripts/tagging/discover_tags.py \
        --source data/sources/my-confluence --sample 100

    # All files
    uv run scripts/tagging/discover_tags.py \
        --source data/sources/my-confluence

Requires `claude` CLI (Max subscription).
"""
import argparse
import json
import logging
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from claude_cli import call_claude, extract_frontmatter, extract_json_array, get_content_excerpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
You are a document tagger for {description}.
Given a document's title, breadcrumb path, and content excerpt, suggest 3-5 short topic tags
that describe what this document is about.

Rules:
- Tags should be lowercase, Norwegian or English, 1-3 words max
- Be specific: prefer domain-specific terms over generic ones
- Include both domain tags (what topic) and type tags (what kind of document)
- Return ONLY a JSON array of strings, e.g. ["topic1", "topic2", "process-type"]
- No explanation, no markdown, just the JSON array

Document:
- Title: {title}
- Breadcrumb: {breadcrumb}
- Content excerpt:
{excerpt}"""


def discover_tags_for_file(md_file: Path, source_dir: Path, description: str,
                           model: str, timeout: int) -> tuple[str, list[str]]:
    """Discover free-form tags for a single file. Returns (rel_path, tags)."""
    rel_path = str(md_file.relative_to(source_dir))
    content = md_file.read_text(encoding='utf-8')
    fields = extract_frontmatter(content)
    title = fields.get('title', md_file.stem)
    breadcrumb = fields.get('breadcrumb', str(md_file.relative_to(source_dir).parent))
    excerpt = get_content_excerpt(content)

    if not excerpt.strip():
        return rel_path, []

    prompt = PROMPT_TEMPLATE.format(description=description, title=title, breadcrumb=breadcrumb, excerpt=excerpt)

    try:
        raw_text = call_claude(prompt, model=model, timeout=timeout)
    except RuntimeError as e:
        logger.error(f"Failed for {rel_path}: {e}")
        return rel_path, []

    tags = extract_json_array(raw_text)
    if tags is None:
        logger.warning(f"Could not extract tags for {rel_path}: {raw_text[:100]}")
        return rel_path, []

    return rel_path, [str(t).lower().strip() for t in tags if isinstance(t, str)]


def main():
    parser = argparse.ArgumentParser(description="Discover natural tags from documents using Claude Haiku")
    parser.add_argument("--source", required=True, help="Source directory with markdown files")
    parser.add_argument("--description", default="a document collection",
                        help="Short description of the collection domain (e.g. 'a Norwegian social security IT system called Melosys')")
    parser.add_argument("--sample", type=int, help="Random sample size (default: all files)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers (default: 10)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per file in seconds")
    parser.add_argument("--output", help="Save raw results to JSON file")
    args = parser.parse_args()

    source_dir = Path(args.source)
    md_files = sorted(source_dir.rglob("*.md"))
    md_files = [f for f in md_files if ".excluded" not in f.parts]

    logger.info(f"Found {len(md_files)} markdown files")

    if args.sample and args.sample < len(md_files):
        md_files = random.sample(md_files, args.sample)
        logger.info(f"Sampled {args.sample} files")

    # Process in parallel
    tag_counter = Counter()
    file_tags: dict[str, list[str]] = {}
    skipped_count = 0
    error_count = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(discover_tags_for_file, f, source_dir, args.description,
                            args.model, args.timeout): f
            for f in md_files
        }

        for future in as_completed(futures):
            completed += 1
            try:
                rel_path, tags = future.result()
                if tags:
                    file_tags[rel_path] = tags
                    tag_counter.update(tags)
                else:
                    skipped_count += 1
            except Exception as e:
                error_count += 1
                logger.error(f"Exception: {e}")

            if completed % 20 == 0:
                logger.info(f"Progress: {completed}/{len(md_files)}")

    # Results
    print(f"\n{'='*60}")
    print(f"Discovery complete: {len(file_tags)} tagged, {skipped_count} skipped, {error_count} errors")
    print(f"Unique tags found: {len(tag_counter)}")
    print(f"{'='*60}")

    print(f"\nTop tags by frequency:")
    for tag, count in tag_counter.most_common(60):
        bar = "█" * min(count, 40)
        print(f"  {tag:30s} {count:4d}  {bar}")

    print(f"\nRare tags (appeared only once):")
    rare = [tag for tag, count in tag_counter.items() if count == 1]
    for tag in sorted(rare):
        print(f"  {tag}")

    if args.output:
        output = {
            "file_count": len(md_files),
            "tagged_count": len(file_tags),
            "error_count": error_count,
            "tag_counts": dict(tag_counter.most_common()),
            "file_tags": file_tags,
        }
        Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
        logger.info(f"Raw results saved to {args.output}")


if __name__ == "__main__":
    main()
