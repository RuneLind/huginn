"""
Reorganize already-downloaded Notion markdown files into correct hierarchy.

Reads notion_id from each .md file's frontmatter, looks up the correct
breadcrumb from the Notion API (page metadata only, no block fetching),
and moves files to the correct folder structure.

Usage:
    export NOTION_TOKEN="ntn_..."
    uv run notion_reorganize_md.py --saveMd "./my-notion"
"""

import os
import shutil
import argparse
import logging

from main.utils.logger import setup_root_logger
from main.utils.filename import sanitize_filename
from main.sources.notion.notion_document_reader import NotionDocumentReader

setup_root_logger()

ap = argparse.ArgumentParser()
ap.add_argument("--saveMd", required=True, help="Directory with existing .md files")
ap.add_argument("--requestDelay", required=False, default=0.1, type=float,
                help="Delay between API requests (default: 0.1, faster since no block fetching)")
ap.add_argument("--dryRun", action="store_true", default=False,
                help="Show what would be moved without actually moving")
args = ap.parse_args()

token = os.environ.get('NOTION_TOKEN')
if not token:
    raise ValueError("NOTION_TOKEN environment variable must be set.")

save_md_path = args.saveMd

# Step 1: Scan all existing .md files and extract notion_id + title
logging.info(f"Scanning {save_md_path} for .md files...")
files_by_id = {}
for root, _dirs, files in os.walk(save_md_path):
    for filename in files:
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(root, filename)
        notion_id = None
        title = None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i > 10:
                        break
                    if line.startswith("notion_id:"):
                        notion_id = line.split(":", 1)[1].strip()
                    elif line.startswith("title:"):
                        title = line.split(":", 1)[1].strip().strip('"')
        except Exception:
            pass
        if notion_id:
            files_by_id[notion_id] = {"path": filepath, "title": title or "Untitled"}

logging.info(f"Found {len(files_by_id)} files with notion_id")

# Step 2: Create a reader just for breadcrumb resolution (no actual reading)
reader = NotionDocumentReader(
    token=token,
    request_delay=args.requestDelay,
)

# Step 3: For each file, look up the page and compute correct breadcrumb
moved = 0
skipped = 0
errors = 0

for i, (notion_id, file_info) in enumerate(files_by_id.items()):
    old_path = file_info["path"]

    try:
        reader.delay()
        page = reader.client.pages.retrieve(page_id=notion_id)
        breadcrumb = reader.build_breadcrumb(page)
        title = reader.get_page_title(page)
    except Exception as e:
        logging.warning(f"Could not retrieve page {notion_id}: {e}")
        errors += 1
        continue

    # Build new path from breadcrumb
    breadcrumb_parts = [p.strip() for p in breadcrumb.split(" -> ")]
    if len(breadcrumb_parts) > 1:
        folder_parts = breadcrumb_parts[:-1]
    else:
        folder_parts = []

    safe_parts = [sanitize_filename(p) for p in folder_parts]
    safe_title = sanitize_filename(title)

    folder = os.path.join(save_md_path, *safe_parts) if safe_parts else save_md_path
    new_path = os.path.join(folder, f"{safe_title}.md")

    if os.path.normpath(old_path) == os.path.normpath(new_path):
        skipped += 1
        continue

    if args.dryRun:
        rel_old = os.path.relpath(old_path, save_md_path)
        rel_new = os.path.relpath(new_path, save_md_path)
        logging.info(f"[DRY RUN] {rel_old} -> {rel_new}")
    else:
        os.makedirs(folder, exist_ok=True)
        # Handle conflicts: if target exists and is a different file
        if os.path.exists(new_path) and os.path.normpath(old_path) != os.path.normpath(new_path):
            # Append notion_id to filename to avoid collision
            base, ext = os.path.splitext(new_path)
            new_path = f"{base} ({notion_id[:8]}){ext}"
        shutil.move(old_path, new_path)

        # Also update the frontmatter breadcrumb if needed
        # (The file content is fine, just the location changed)

    moved += 1

    if (i + 1) % 100 == 0:
        logging.info(f"Progress: {i + 1}/{len(files_by_id)} pages processed")

# Step 4: Clean up empty directories
if not args.dryRun:
    for root, dirs, files in os.walk(save_md_path, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception:
                pass

action = "Would move" if args.dryRun else "Moved"
logging.info(f"Done. {action} {moved} files, skipped {skipped} (already correct), errors {errors}")
