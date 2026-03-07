import asyncio
import json
import os
import argparse
from datetime import datetime, timezone
from playwright.async_api import async_playwright
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString
import re
from typing import Dict, List, Optional


class HierarchicalConfluenceFetcher:
    def __init__(self, base_url: str, output_dir: str):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.page_hierarchy = {}  # Map page_id to parent info

    async def fetch_all_pages_with_hierarchy(self, space_key: str, start_from_time: Optional[str] = None):
        """Fetch all pages from a Confluence space with hierarchy information.

        Args:
            space_key: Confluence space key (e.g. 'MYSPACE')
            start_from_time: ISO datetime string — only fetch pages modified on or after this date
        """

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)

            storage_state = None
            auth_file = Path(__file__).parent.parent / "auth" / "confluence_auth.json"
            if auth_file.exists():
                storage_state = str(auth_file)

            context = await browser.new_context(storage_state=storage_state)
            page = await context.new_page()

            space_url = f"{self.base_url}/spaces/{space_key}/overview"
            await page.goto(space_url)

            try:
                await page.wait_for_url("**/spaces/**", timeout=120000)
                print("✅ Authenticated successfully")
            except Exception:
                print("❌ Authentication timeout")
                await browser.close()
                return

            # Save auth state for future use
            auth_file.parent.mkdir(parents=True, exist_ok=True)
            await context.storage_state(path=str(auth_file))

            # Build CQL query
            cql = f"space = '{space_key}' AND type = page"
            if start_from_time:
                # Confluence CQL expects YYYY-MM-DD format for date comparisons
                cql_date = start_from_time[:10]  # Extract date part from ISO string
                cql += f" AND lastModified >= '{cql_date}'"
                print(f"📅 Filtering: lastModified >= {cql_date}")

            all_pages = []
            start = 0
            limit = 50
            total_fetched = 0

            while True:
                api_url = f"{self.base_url}/rest/api/content/search"

                print(f"\n📥 Fetching pages {start} to {start + limit}...")

                try:
                    response = await page.request.get(
                        api_url,
                        params={
                            'cql': cql,
                            'limit': limit,
                            'start': start,
                            'expand': 'body.storage,version,space,ancestors'
                        }
                    )

                    if response.status != 200:
                        print(f"❌ API returned status {response.status}")
                        break

                    text = await response.text()
                    data = json.loads(text)
                    results = data.get('results', [])

                    if not results:
                        print("✅ No more pages to fetch")
                        break

                    all_pages.extend(results)
                    total_fetched += len(results)

                    print(f"   Got {len(results)} pages (total: {total_fetched})")

                    if len(results) < limit:
                        break

                    start += limit

                except Exception as e:
                    print(f"❌ Error fetching pages: {e}")
                    break

            await browser.close()

            print(f"\n✅ Total pages fetched: {len(all_pages)}")
            return all_pages

    def build_hierarchy(self, pages: List[Dict]) -> Dict:
        """Build a hierarchy tree from pages with ancestors"""

        hierarchy = {}

        for page in pages:
            page_id = page['id']
            title = page['title']
            ancestors = page.get('ancestors', [])

            # Get parent path
            parent_path = []
            for ancestor in ancestors:
                ancestor_title = ancestor.get('title', 'Unknown')
                # Clean title for folder name
                clean_title = self.sanitize_filename(ancestor_title)
                parent_path.append(clean_title)

            hierarchy[page_id] = {
                'title': title,
                'parent_path': parent_path,
                'page': page
            }

        return hierarchy

    def sanitize_filename(self, name: str) -> str:
        """Create safe filename/foldername from title"""
        # Remove invalid characters
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)
        # Trim and limit length
        name = name.strip()[:100]
        return name

    def extract_cell_text(self, cell) -> str:
        """Extract text from table cell, handling Confluence emoticons"""

        cell_str = str(cell)
        emoticon_pattern = r'<ac:emoticon ac:name="([^"]+)"'
        emoticon_matches = re.findall(emoticon_pattern, cell_str)

        if emoticon_matches:
            emoticon_name = emoticon_matches[0]
            emoticon_map = {
                'tick': '✓',
                'cross': '❌',
                'minus': '⊝',
                'warning': '⚠️',
                'information': 'ℹ️',
                'question': '❓',
            }
            if emoticon_name in emoticon_map:
                return emoticon_map[emoticon_name]

        images = cell.find_all('img')
        for img in images:
            alt = img.get('alt', '').lower()
            if 'check' in alt or 'tick' in alt:
                return '✓'
            elif 'cross' in alt or 'error' in alt:
                return '❌'

        text = cell.get_text(separator=' ', strip=True)

        if not text or text.strip() == '':
            return ' '

        text = text.replace('|', '\\|').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def convert_table_to_markdown(self, table_element) -> str:
        """Convert HTML table to Markdown table"""

        tbody = table_element.find('tbody')
        thead = table_element.find('thead')

        markdown_rows = []

        # Process header
        if thead:
            header_rows = thead.find_all('tr')
            for row in header_rows:
                cells = row.find_all(['th', 'td'])
                if cells:
                    cell_texts = [self.extract_cell_text(cell) for cell in cells]
                    markdown_rows.append('| ' + ' | '.join(cell_texts) + ' |')
                    markdown_rows.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')

        # Process body rows
        rows_to_process = []
        if tbody:
            rows_to_process = tbody.find_all('tr')
        else:
            rows_to_process = table_element.find_all('tr', recursive=False)

        # If no thead, ALWAYS treat first row as header
        start_idx = 0
        if not thead and rows_to_process:
            first_row = rows_to_process[0]
            cells = first_row.find_all(['td', 'th'])

            # Always treat first row as header in Markdown tables
            if cells:
                cell_texts = [self.extract_cell_text(cell) for cell in cells]
                markdown_rows.append('| ' + ' | '.join(cell_texts) + ' |')
                markdown_rows.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')
                start_idx = 1

        # Process data rows
        for row in rows_to_process[start_idx:]:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue

            cell_texts = [self.extract_cell_text(cell) for cell in cells]
            markdown_rows.append('| ' + ' | '.join(cell_texts) + ' |')

        return '\n'.join(markdown_rows)

    def process_element(self, element):
        """Recursively process element to extract text with formatting"""
        if isinstance(element, NavigableString):
            return str(element)

        if element.name in ['script', 'style']:
            return ''

        if element.name in ['strong', 'b']:
            inner_text = ''.join([self.process_element(child) for child in element.children])
            inner_text = inner_text.strip()
            if inner_text:
                return f'**{inner_text}**'
            return ''

        if element.name in ['em', 'i']:
            inner_text = ''.join([self.process_element(child) for child in element.children])
            inner_text = inner_text.strip()
            if inner_text:
                return f'*{inner_text}*'
            return ''

        if element.name == 'code':
            return f'`{element.get_text()}`'

        if element.name == 'a':
            href = element.get('href', '')
            text = element.get_text(strip=True)
            if href and text:
                if href.startswith('/x/'):
                    href = f"{self.base_url}{href}"
                return f'[{text}]({href})'
            return text

        return ''.join([self.process_element(child) for child in element.children])

    def html_to_markdown(self, html_content: str) -> str:
        """Convert Confluence HTML storage format to Markdown"""

        soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')

        # Handle Confluence layout sections - unwrap them to process contents normally
        for layout in soup.find_all(['ac:layout', 'ac:layout-section', 'ac:layout-cell']):
            layout.unwrap()

        # Remove Confluence macros but keep layout structure
        for tag in soup.find_all(['ac:structured-macro', 'ac:parameter', 'ri:attachment', 'ri:page']):
            tag.decompose()

        markdown_parts = []

        body = soup.find('body') or soup
        elements = list(body.children) if hasattr(body, 'children') else [body]

        for element in elements:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    markdown_parts.append(f'{text}\n\n')
                continue

            if not hasattr(element, 'name'):
                continue

            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                text = element.get_text(strip=True)
                markdown_parts.append(f'\n{"#" * level} {text}\n')

            elif element.name == 'table':
                markdown_table = self.convert_table_to_markdown(element)
                if markdown_table:
                    markdown_parts.append(f'\n{markdown_table}\n')

            elif element.name == 'pre':
                code_text = element.get_text()
                markdown_parts.append(f'\n```\n{code_text}\n```\n')

            elif element.name == 'ul':
                items = element.find_all('li', recursive=False)
                for item in items:
                    text = self.process_element(item).strip()
                    if text:
                        markdown_parts.append(f'- {text}\n')
                markdown_parts.append('\n')

            elif element.name == 'ol':
                items = element.find_all('li', recursive=False)
                for idx, item in enumerate(items, 1):
                    text = self.process_element(item).strip()
                    if text:
                        markdown_parts.append(f'{idx}. {text}\n')
                markdown_parts.append('\n')

            elif element.name == 'p':
                text = self.process_element(element).strip()
                if text:
                    markdown_parts.append(f'{text}\n\n')

            elif element.name == 'div':
                text = self.process_element(element).strip()
                if text:
                    markdown_parts.append(f'{text}\n\n')

            else:
                text = self.process_element(element).strip()
                if text:
                    markdown_parts.append(f'{text}\n\n')

        markdown = ''.join(markdown_parts)

        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        markdown = re.sub(r'\*\*\*\*(\w+)\*\*\*\*', r'**\1**', markdown)
        markdown = re.sub(r'\*\*(\w+)\*\*\*\*', r'**\1**', markdown)

        return markdown.strip()

    @staticmethod
    def _set_file_mtime(file_path: Path, iso_time: str):
        """Set the file's modification time to match the Confluence page's modifiedTime.

        This ensures that files_document_reader picks up the actual Confluence
        modification time (not the time the file was written to disk), which is
        critical for correct incremental update cutoff calculations.
        """
        if not iso_time:
            return
        try:
            ts = datetime.fromisoformat(iso_time).timestamp()
            os.utime(file_path, (ts, ts))
        except (ValueError, OSError):
            pass  # keep OS mtime if parsing fails

    def _build_frontmatter(self, title: str, page_id: str, space_key: str,
                            modified_time: str, breadcrumb: str) -> str:
        """Build YAML frontmatter for a Confluence markdown page."""
        return f"""---
title: {title}
page_id: {page_id}
space: {space_key}
modifiedTime: {modified_time}
breadcrumb: {breadcrumb}
url: {self.base_url}/spaces/{space_key}/pages/{page_id}
---

"""

    def save_pages_with_hierarchy(self, pages: List[Dict], format: str = 'both'):
        """Save pages preserving Confluence hierarchy"""

        # Build hierarchy tree
        print("\n🌳 Building hierarchy tree...")
        hierarchy = self.build_hierarchy(pages)

        json_base = self.output_dir / 'json'
        md_base = self.output_dir / 'markdown'

        json_base.mkdir(exist_ok=True)
        md_base.mkdir(exist_ok=True)

        for page_data in pages:
            page_id = page_data['id']
            title = page_data['title']
            html_content = page_data.get('body', {}).get('storage', {}).get('value', '')
            space_key = page_data.get('space', {}).get('key', 'UNKNOWN')
            modified_time = page_data.get('version', {}).get('when', '')

            # Get hierarchy info
            page_info = hierarchy.get(page_id, {})
            parent_path = page_info.get('parent_path', [])

            # Create safe filename
            safe_title = self.sanitize_filename(title)

            # Build directory path
            if parent_path:
                json_dir = json_base / Path(*parent_path)
                md_dir = md_base / Path(*parent_path)
            else:
                json_dir = json_base
                md_dir = md_base

            # Create directories
            if format in ['json', 'both']:
                json_dir.mkdir(parents=True, exist_ok=True)

            if format in ['markdown', 'both']:
                md_dir.mkdir(parents=True, exist_ok=True)

            # Save JSON
            if format in ['json', 'both']:
                json_file = json_dir / f"{page_id}_{safe_title}.json"
                json_file.write_text(
                    json.dumps({
                        'id': page_id,
                        'title': title,
                        'space': space_key,
                        'html_content': html_content,
                        'url': f"{self.base_url}/spaces/{space_key}/pages/{page_id}",
                        'parent_path': parent_path
                    }, indent=2, ensure_ascii=False),
                    encoding='utf-8'
                )

            # Save Markdown
            if format in ['markdown', 'both']:
                markdown_content = self.html_to_markdown(html_content)
                breadcrumb = ' > '.join(parent_path) if parent_path else space_key
                frontmatter = self._build_frontmatter(title, page_id, space_key, modified_time, breadcrumb)
                md_file = md_dir / f"{safe_title}.md"
                md_file.write_text(frontmatter + markdown_content, encoding='utf-8')
                self._set_file_mtime(md_file, modified_time)

            # Show hierarchy in output
            indent = "  " * len(parent_path)
            print(f"{indent}📄 {title}")

        print(f"\n📁 Files saved to: {self.output_dir}")
        if format in ['json', 'both']:
            print(f"   JSON: {json_base}")
        if format in ['markdown', 'both']:
            print(f"   Markdown: {md_base}")

    @staticmethod
    def scan_existing_page_ids(md_dir: str) -> set:
        """Scan .md files in a directory and return a set of page_ids from frontmatter."""
        page_ids = set()
        md_path = Path(md_dir)
        if not md_path.exists():
            return page_ids

        for md_file in md_path.rglob("*.md"):
            # Skip .excluded directory
            if ".excluded" in md_file.parts:
                continue
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    in_fm = False
                    for line in f:
                        if line.strip() == "---" and not in_fm:
                            in_fm = True
                            continue
                        if line.strip() == "---" and in_fm:
                            break
                        if in_fm and line.startswith("page_id:"):
                            page_id = line.partition(":")[2].strip().strip('"')
                            if page_id:
                                page_ids.add(page_id)
                            break
            except Exception:
                pass

        return page_ids

    @staticmethod
    def load_exclude_manifest(manifest_path: str) -> set:
        """Load excluded page_ids from a manifest file."""
        path = Path(manifest_path)
        if not path.exists():
            return set()
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        return {e["page_id"] for e in entries if e.get("page_id")}

    def save_pages_as_markdown(self, pages: List[Dict], save_md_path: str,
                                skip_page_ids: Optional[set] = None):
        """Save pages as markdown directly to a directory (no json/ or markdown/ subdirs).

        This is the format used by the curated content directories and the cleanup script.
        """
        md_base = Path(save_md_path)
        md_base.mkdir(parents=True, exist_ok=True)

        hierarchy = self.build_hierarchy(pages)
        saved = 0
        skipped = 0

        for page_data in pages:
            page_id = page_data['id']

            if skip_page_ids and page_id in skip_page_ids:
                skipped += 1
                continue

            title = page_data['title']
            html_content = page_data.get('body', {}).get('storage', {}).get('value', '')
            space_key = page_data.get('space', {}).get('key', 'UNKNOWN')
            modified_time = page_data.get('version', {}).get('when', '')

            page_info = hierarchy.get(page_id, {})
            parent_path = page_info.get('parent_path', [])

            safe_title = self.sanitize_filename(title)

            if parent_path:
                md_dir = md_base / Path(*parent_path)
            else:
                md_dir = md_base

            md_dir.mkdir(parents=True, exist_ok=True)

            markdown_content = self.html_to_markdown(html_content)
            breadcrumb = ' > '.join(parent_path) if parent_path else space_key
            frontmatter = self._build_frontmatter(title, page_id, space_key, modified_time, breadcrumb)
            md_file = md_dir / f"{safe_title}.md"
            md_file.write_text(frontmatter + markdown_content, encoding='utf-8')
            self._set_file_mtime(md_file, modified_time)
            saved += 1

            indent = "  " * len(parent_path)
            print(f"   Saved: {indent}{title}")

        print(f"\n📁 Saved {saved} pages to {md_base}")
        if skipped:
            print(f"   Skipped {skipped} (already existing or excluded)")

    def save_fetch_metadata(self, space_key: str, total_pages: int):
        """Save metadata about the fetch for update detection"""
        metadata = {
            "space_key": space_key,
            "base_url": self.base_url,
            "fetch_time": datetime.now(timezone.utc).isoformat(),
            "total_pages": total_pages,
            "output_dir": str(self.output_dir)
        }

        metadata_file = self.output_dir / "fetch_metadata.json"
        metadata_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        print(f"\n📋 Metadata saved to: {metadata_file}")
        return metadata


def get_default_output_dir() -> str:
    """Get the default output directory relative to project root"""
    # Navigate from script location to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # scripts/confluence/fetchers -> root
    return str(project_root / "data" / "downloaded" / "confluence_hierarchical")


async def main():
    parser = argparse.ArgumentParser(
        description="Fetch Confluence pages with hierarchical structure preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch MYSPACE space (default) — saves json + markdown to output dir
  uv run confluence_fetcher_hierarchical.py --space MYSPACE

  # Save markdown directly to curated directory (no json/ or markdown/ subdirs)
  uv run confluence_fetcher_hierarchical.py --space MYSPACE --saveMd ./data/sources/my-confluence

  # Incremental download: only pages modified since cutoff, skip existing
  uv run confluence_fetcher_hierarchical.py --space MYSPACE --saveMd ./data/sources/my-confluence \\
      --skipExisting --startFromTime "2025-12-01T00:00:00"

  # Skip excluded pages from manifest
  uv run confluence_fetcher_hierarchical.py --space MYSPACE --saveMd ./data/sources/my-confluence \\
      --skipExisting --excludeManifest ./data/sources/my-confluence/.excluded/excluded_manifest.json
        """
    )

    parser.add_argument(
        "--space", "-s",
        default="MYSPACE",
        help="Confluence space key (default: MYSPACE)"
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for json+markdown (default: ./data/downloaded/confluence_hierarchical)"
    )

    parser.add_argument(
        "--saveMd",
        default=None,
        help="Save markdown directly to this directory (no json/ or markdown/ subdirs). "
             "When set, --output and --format are ignored."
    )

    parser.add_argument(
        "--base-url", "-u",
        required=True,
        help="Confluence base URL (e.g. https://confluence.example.com)"
    )

    parser.add_argument(
        "--format", "-f",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both). Ignored when --saveMd is used."
    )

    parser.add_argument(
        "--skipExisting",
        action="store_true",
        default=False,
        help="Skip pages that already have a .md file on disk (by page_id in frontmatter)"
    )

    parser.add_argument(
        "--excludeManifest",
        default=None,
        help="Path to excluded_manifest.json — skip page_ids listed there"
    )

    parser.add_argument(
        "--startFromTime",
        default=None,
        help="Only fetch pages modified on or after this ISO datetime (e.g. 2025-12-01T00:00:00)"
    )

    args = parser.parse_args()

    # Determine output mode
    save_md = args.saveMd
    output_dir = args.output if args.output else get_default_output_dir()

    print(f"🚀 Confluence Hierarchical Fetcher")
    print(f"   Space: {args.space}")
    print(f"   Base URL: {args.base_url}")
    if save_md:
        print(f"   Save markdown to: {save_md}")
    else:
        print(f"   Output: {output_dir}")
        print(f"   Format: {args.format}")
    if args.startFromTime:
        print(f"   Start from: {args.startFromTime}")
    if args.skipExisting:
        print(f"   Skip existing: yes")
    if args.excludeManifest:
        print(f"   Exclude manifest: {args.excludeManifest}")
    print()

    # Use saveMd dir or output dir for fetcher base
    fetcher_dir = save_md if save_md else output_dir
    fetcher = HierarchicalConfluenceFetcher(
        base_url=args.base_url,
        output_dir=fetcher_dir
    )

    pages = await fetcher.fetch_all_pages_with_hierarchy(
        args.space,
        start_from_time=args.startFromTime,
    )

    if not pages:
        print("❌ No pages were fetched")
        return

    if save_md:
        # Build set of page_ids to skip
        skip_ids = set()
        if args.skipExisting:
            existing = HierarchicalConfluenceFetcher.scan_existing_page_ids(save_md)
            skip_ids.update(existing)
            print(f"📋 Found {len(existing)} existing pages on disk")

        if args.excludeManifest:
            excluded = HierarchicalConfluenceFetcher.load_exclude_manifest(args.excludeManifest)
            skip_ids.update(excluded)
            print(f"📋 Loaded {len(excluded)} excluded page_ids from manifest")

        fetcher.save_pages_as_markdown(pages, save_md, skip_page_ids=skip_ids)
        fetcher.save_fetch_metadata(args.space, len(pages))
        print(f"\n🎉 Done! {len(pages)} pages fetched from Confluence")
    else:
        fetcher.save_pages_with_hierarchy(pages, format=args.format)
        fetcher.save_fetch_metadata(args.space, len(pages))
        print(f"\n🎉 Successfully processed {len(pages)} pages with hierarchy!")


if __name__ == "__main__":
    asyncio.run(main())
