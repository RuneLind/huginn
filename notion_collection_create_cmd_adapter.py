import os
import json
import argparse
import logging
from datetime import datetime, timezone

from main.utils.logger import setup_root_logger
from main.utils.filename import sanitize_filename
from main.sources.notion.notion_document_reader import NotionDocumentReader
from main.sources.notion.notion_document_converter import NotionDocumentConverter
from main.sources.notion.notion_block_to_markdown import convert_blocks_to_markdown, extract_page_properties, extract_page_properties_structured
from main.factories.create_collection_factory import create_collection_creator

setup_root_logger()

ap = argparse.ArgumentParser()
ap.add_argument("-collection", "--collection", required=False, default=None, help="Collection name (will be used as root folder name)")
ap.add_argument("-downloadOnly", "--downloadOnly", action="store_true", required=False, default=False, help="Only download markdown files (requires --saveMd, skips indexing)")
ap.add_argument("-skipExisting", "--skipExisting", action="store_true", required=False, default=False, help="Skip pages that already have a .md file on disk (useful for resuming)")
ap.add_argument("-rootPageId", "--rootPageId", required=False, default=None, help="Optional root page ID to scope to a subtree")
ap.add_argument("-saveMd", "--saveMd", required=False, default=None, help="Directory path to save markdown files (e.g., ./notion_pages)")
ap.add_argument("-requestDelay", "--requestDelay", required=False, default=0.35, type=float, help="Delay between API requests in seconds (default: 0.35)")
ap.add_argument("-indexers", "--indexers", required=False, default=["indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base", "indexer_BM25"], help="List of indexer names", nargs='+')
ap.add_argument("-excludeManifest", "--excludeManifest", required=False, default=None, help="Path to excluded_manifest.json (skip excluded pages unless updated in Notion)")
ap.add_argument("-startFromTime", "--startFromTime", required=False, default=None, help="ISO datetime cutoff — only fetch pages modified after this time (e.g., 2026-02-15T00:00:00)")
args = vars(ap.parse_args())

token = os.environ.get('NOTION_TOKEN')
if not token:
    raise ValueError("NOTION_TOKEN environment variable must be set. Create an integration at https://www.notion.so/my-integrations")


def _scan_existing_page_ids(directory):
    """Scan .md files in directory for notion_id in YAML frontmatter."""
    page_ids = set()
    if not os.path.isdir(directory):
        return page_ids

    for root, _dirs, files in os.walk(directory):
        _dirs[:] = [d for d in _dirs if d != ".excluded"]
        for filename in files:
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    # Only read the first 10 lines (frontmatter)
                    for i, line in enumerate(f):
                        if i > 10:
                            break
                        if line.startswith("notion_id:"):
                            page_id = line.split(":", 1)[1].strip()
                            if page_id:
                                page_ids.add(page_id)
                            break
            except Exception:
                pass

    return page_ids


skip_page_ids = set()
if args['skipExisting'] and args['saveMd']:
    skip_page_ids = _scan_existing_page_ids(args['saveMd'])
    if skip_page_ids:
        logging.info(f"Found {len(skip_page_ids)} already downloaded pages, will skip them")

start_from_time = None
if args['startFromTime']:
    start_from_time = datetime.fromisoformat(args['startFromTime'])
    if start_from_time.tzinfo is None:
        start_from_time = start_from_time.replace(tzinfo=timezone.utc)
    logging.info(f"Incremental mode: only fetching pages modified after {start_from_time.isoformat()}")

exclude_unless_updated = None
if args['excludeManifest']:
    manifest_path = args['excludeManifest']
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        exclude_unless_updated = {
            e["notion_id"]: e["last_edited_time"]
            for e in entries if e.get("notion_id")
        }
        logging.info(f"Loaded exclude manifest with {len(exclude_unless_updated)} pages")
    else:
        logging.warning(f"Exclude manifest not found: {manifest_path}")

notion_document_reader = NotionDocumentReader(
    token=token,
    root_page_id=args['rootPageId'],
    request_delay=args['requestDelay'],
    skip_page_ids=skip_page_ids,
    exclude_unless_updated=exclude_unless_updated,
    start_from_time=start_from_time,
)

save_md_path = args['saveMd']


def _build_filepath(breadcrumb, title):
    breadcrumb_parts = [p.strip() for p in breadcrumb.split(" -> ")]
    if len(breadcrumb_parts) > 1:
        folder_parts = breadcrumb_parts[:-1]
    else:
        folder_parts = []

    safe_parts = [sanitize_filename(p) for p in folder_parts]
    safe_title = sanitize_filename(title)

    folder = os.path.join(save_md_path, *safe_parts) if safe_parts else save_md_path
    return os.path.join(folder, f"{safe_title}.md")


def _save_markdown(document, converted_result):
    """Save converted document as markdown file with frontmatter."""
    breadcrumb = converted_result.get("breadcrumb", "")
    title = converted_result.get("title", "Untitled")
    page_id = converted_result["id"]
    last_edited = converted_result.get("modifiedTime", "")
    url = converted_result.get("url", "")

    filepath = _build_filepath(breadcrumb, title)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    page = document["page"]
    props_structured = extract_page_properties_structured(page.get("properties", {}))
    properties_md = extract_page_properties(page.get("properties", {}))
    markdown_body = convert_blocks_to_markdown(document["blocks"])

    frontmatter_lines = [
        "---",
        f"title: \"{title}\"",
        f"url: {url}",
        f"last_edited_time: {last_edited}",
        f"notion_id: {page_id}",
    ]
    if props_structured.get("Tags"):
        frontmatter_lines.append(f"tags: {props_structured['Tags']}")
    if props_structured.get("Created by"):
        frontmatter_lines.append(f"created_by: {props_structured['Created by']}")
    if props_structured.get("Status"):
        frontmatter_lines.append(f"status: {props_structured['Status']}")
    frontmatter_lines.append("---\n")
    frontmatter = "\n".join(frontmatter_lines) + "\n"

    body_parts = [p for p in [properties_md, markdown_body] if p]
    body = "\n\n".join(body_parts)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(frontmatter + body)

    if last_edited:
        try:
            ts = datetime.fromisoformat(last_edited).timestamp()
            os.utime(filepath, (ts, ts))
        except (ValueError, OSError):
            pass

    logging.info(f"Saved: {filepath}")


on_convert = _save_markdown if save_md_path else None
notion_document_converter = NotionDocumentConverter(on_convert=on_convert)


download_only = args['downloadOnly']

if download_only:
    if not save_md_path:
        raise ValueError("--saveMd is required when using --downloadOnly")

    logging.info(f"Download-only mode: saving markdown files to {save_md_path}")
    count = 0
    for document in notion_document_reader.read_all_documents():
        notion_document_converter.convert(document)
        count += 1
    logging.info(f"Done. Downloaded {count} pages to {save_md_path}")
else:
    if not args['collection']:
        raise ValueError("--collection is required unless using --downloadOnly")

    notion_collection_creator = create_collection_creator(
        collection_name=args['collection'],
        indexers=args['indexers'],
        document_reader=notion_document_reader,
        document_converter=notion_document_converter,
    )

    notion_collection_creator.run()
