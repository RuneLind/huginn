import time
import logging
from datetime import datetime

from notion_client import Client

logger = logging.getLogger(__name__)


class NotionDocumentReader:
    def __init__(self,
                 token,
                 root_page_id=None,
                 request_delay=0.35,
                 start_from_time=None,
                 max_block_depth=10,
                 skip_page_ids=None,
                 exclude_unless_updated=None):
        self.client = Client(auth=token)
        self.root_page_id = root_page_id
        self.request_delay = request_delay
        self.start_from_time = start_from_time
        self.max_block_depth = max_block_depth
        self.skip_page_ids = skip_page_ids or set()
        self.exclude_unless_updated = exclude_unless_updated or {}
        self._parent_cache = {}
        self._data_sources_loaded = False

    def read_all_documents(self):
        for page in self._iterate_pages():
            page_id = page["id"]
            if page_id in self.skip_page_ids:
                title = self.get_page_title(page)
                logger.info(f"Skipping already processed page: '{title}' ({page_id})")
                continue
            if page_id in self.exclude_unless_updated:
                if not self._has_been_updated_since_exclusion(page, page_id):
                    continue
            try:
                blocks = self._fetch_all_blocks(page_id)
                breadcrumb = self.build_breadcrumb(page)
                self._resolve_relation_titles(page)

                yield {
                    "page": page,
                    "blocks": blocks,
                    "breadcrumb": breadcrumb,
                }
            except Exception as e:
                title = self.get_page_title(page)
                logger.error(f"Error reading page '{title}' ({page_id}): {e}")

    def get_number_of_documents(self):
        # Notion search API does not provide a total count.
        # Return 0 so the progress bar shows items processed without a total.
        return 0

    def get_reader_details(self):
        return {
            "type": "notion",
            "rootPageId": self.root_page_id,
            "requestDelay": self.request_delay,
        }

    def _iterate_pages(self):
        if self.root_page_id:
            yield from self._iterate_child_pages(self.root_page_id)
        else:
            yield from self._search_all_pages()

    def _search_all_pages(self):
        start_cursor = None
        while True:
            self.delay()
            kwargs = {
                "filter": {"value": "page", "property": "object"},
                "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                "page_size": 100,
            }
            if start_cursor:
                kwargs["start_cursor"] = start_cursor

            result = self._api_call_with_retry(lambda kw=kwargs: self.client.search(**kw))
            if result is None:
                return

            for page in result.get("results", []):
                if self.start_from_time and self._is_before_cutoff(page):
                    return
                yield page

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

    def _iterate_child_pages(self, page_id):
        # First yield the root page itself
        self.delay()
        root_page = self.client.pages.retrieve(page_id=page_id)

        if self.start_from_time and self._is_before_cutoff(root_page):
            pass  # still recurse children — they may have been edited more recently
        else:
            yield root_page

        # Then recursively find child pages via blocks
        yield from self._find_child_pages_in_blocks(page_id)

    def _find_child_pages_in_blocks(self, block_id, depth=0):
        if depth >= self.max_block_depth:
            return

        start_cursor = None
        while True:
            self.delay()
            kwargs = {"block_id": block_id, "page_size": 100}
            if start_cursor:
                kwargs["start_cursor"] = start_cursor

            result = self._api_call_with_retry(lambda kw=kwargs: self.client.blocks.children.list(**kw))
            if result is None:
                return

            for block in result.get("results", []):
                if block.get("type") == "child_page":
                    child_page_id = block["id"]
                    yield from self._iterate_child_pages(child_page_id)
                elif block.get("type") == "child_database":
                    yield from self._iterate_database_pages(block["id"])
                elif block.get("has_children"):
                    yield from self._find_child_pages_in_blocks(block["id"], depth + 1)

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

    def _iterate_database_pages(self, database_id):
        start_cursor = None
        while True:
            self.delay()
            kwargs = {"database_id": database_id, "page_size": 100}
            if start_cursor:
                kwargs["start_cursor"] = start_cursor

            result = self._api_call_with_retry(lambda kw=kwargs: self.client.databases.query(**kw))
            if result is None:
                return

            for page in result.get("results", []):
                if self.start_from_time and self._is_before_cutoff(page):
                    continue
                yield page

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

    def _fetch_all_blocks(self, page_id, depth=0):
        if depth >= self.max_block_depth:
            return []

        blocks = []
        start_cursor = None

        while True:
            self.delay()
            kwargs = {"block_id": page_id, "page_size": 100}
            if start_cursor:
                kwargs["start_cursor"] = start_cursor

            result = self._api_call_with_retry(lambda kw=kwargs: self.client.blocks.children.list(**kw))
            if result is None:
                break

            for block in result.get("results", []):
                if block.get("has_children"):
                    block["children"] = self._fetch_all_blocks(block["id"], depth + 1)
                blocks.append(block)

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

        return blocks

    def build_breadcrumb(self, page):
        parts = []
        current = page
        max_depth = 15

        for _ in range(max_depth):
            parent = current.get("parent", {})
            parent_type = parent.get("type")

            if parent_type == "page_id":
                parent_id = parent["page_id"]
                parent_page = self._get_cached_page(parent_id)
                if parent_page:
                    parts.append(self.get_page_title(parent_page))
                    current = parent_page
                else:
                    break
            elif parent_type == "database_id":
                db_id = parent["database_id"]
                db_info = self._get_database_info(db_id)
                if db_info:
                    title, db_parent = db_info
                    if title:
                        parts.append(title)
                    # Continue walking up through the database's parent
                    current = {"parent": db_parent} if db_parent else {}
                    if not db_parent:
                        break
                else:
                    break
            elif parent_type == "data_source_id":
                ds_id = parent["data_source_id"]
                ds_info = self._get_data_source_info(ds_id)
                if ds_info:
                    title, ds_parent = ds_info
                    if title:
                        parts.append(title)
                    current = {"parent": ds_parent} if ds_parent else {}
                    if not ds_parent:
                        break
                else:
                    break
            elif parent_type == "block_id":
                # Block parents can be pages or blocks inside pages
                block_id = parent["block_id"]
                block_parent = self._resolve_block_parent(block_id)
                if block_parent:
                    current = {"parent": block_parent}
                else:
                    break
            else:
                break

        parts.reverse()
        parts.append(self.get_page_title(page))
        return " -> ".join(parts)

    def _get_cached_page(self, page_id):
        if page_id in self._parent_cache:
            return self._parent_cache[page_id]

        try:
            self.delay()
            page = self.client.pages.retrieve(page_id=page_id)
            self._parent_cache[page_id] = page
            return page
        except Exception as e:
            logger.warning(f"Could not retrieve parent page {page_id}: {e}")
            self._parent_cache[page_id] = None
            return None

    def _resolve_block_parent(self, block_id):
        """Try to resolve a block_id to its parent. Returns a parent dict or None."""
        cache_key = f"block:{block_id}"
        if cache_key in self._parent_cache:
            return self._parent_cache[cache_key]

        try:
            self.delay()
            block = self.client.blocks.retrieve(block_id=block_id)
            parent = block.get("parent", {})
            self._parent_cache[cache_key] = parent
            return parent
        except Exception as e:
            logger.debug(f"Block retrieve failed for {block_id}, trying as page: {e}")
            # block_id might actually be a page
            try:
                page = self._get_cached_page(block_id)
                if page:
                    parts_parent = page.get("parent", {})
                    self._parent_cache[cache_key] = parts_parent
                    return {"type": "page_id", "page_id": block_id}
            except Exception as e2:
                logger.debug(f"Page retrieve also failed for {block_id}: {e2}")
            self._parent_cache[cache_key] = None
            return None

    def _get_database_info(self, database_id):
        """Returns (title, parent_dict) or None."""
        cache_key = f"db:{database_id}"
        if cache_key in self._parent_cache:
            return self._parent_cache[cache_key]

        try:
            self.delay()
            db = self.client.databases.retrieve(database_id=database_id)
            title_parts = db.get("title", [])
            title = "".join(t.get("plain_text", "") for t in title_parts)
            parent = db.get("parent", {})
            result = (title, parent)
            self._parent_cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Could not retrieve database {database_id}: {e}")
            self._parent_cache[cache_key] = None
            return None

    def _get_data_source_info(self, data_source_id):
        """Returns (title, database_parent_dict) or None."""
        cache_key = f"ds:{data_source_id}"
        if cache_key in self._parent_cache:
            return self._parent_cache[cache_key]

        try:
            # Data sources aren't retrievable by ID — bulk load all on first hit
            if not self._data_sources_loaded:
                self._load_all_data_sources()

            return self._parent_cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Could not retrieve data source {data_source_id}: {e}")
            self._parent_cache[cache_key] = None
            return None

    def _load_all_data_sources(self):
        """Load all data sources into cache in one pass."""
        self._data_sources_loaded = True
        start_cursor = None
        count = 0
        while True:
            self.delay()
            kwargs = {
                "filter": {"value": "data_source", "property": "object"},
                "page_size": 100,
            }
            if start_cursor:
                kwargs["start_cursor"] = start_cursor

            result = self.client.search(**kwargs)

            for ds in result.get("results", []):
                ds_id = ds["id"]
                title_parts = ds.get("title", [])
                title = "".join(t.get("plain_text", "") for t in title_parts)
                # data sources have database_parent for the real hierarchy
                db_parent = ds.get("database_parent", {})
                cache_key = f"ds:{ds_id}"
                self._parent_cache[cache_key] = (title, db_parent)
                count += 1

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

        logger.info(f"Loaded {count} data sources for hierarchy resolution")

    def _has_been_updated_since_exclusion(self, page, page_id):
        """Check if page has been updated since it was excluded. Returns True if re-fetch needed."""
        excluded_time = self.exclude_unless_updated.get(page_id, "")
        page_time = page.get("last_edited_time", "")
        title = self.get_page_title(page)

        if not excluded_time or not page_time:
            logger.info(f"Re-fetching previously excluded page (missing timestamps): '{title}' ({page_id})")
            return True

        if page_time > excluded_time:
            logger.info(f"Re-fetching previously excluded page (updated): '{title}' ({page_id})")
            return True

        logger.debug(f"Skipping excluded page (not updated): '{title}' ({page_id})")
        return False

    def _is_before_cutoff(self, page):
        last_edited = page.get("last_edited_time", "")
        if not last_edited:
            return False
        page_time = datetime.fromisoformat(last_edited.replace("Z", "+00:00"))
        return page_time < self.start_from_time

    @staticmethod
    def get_page_title(page):
        properties = page.get("properties", {})

        # Check "title" type property (standard pages)
        for prop in properties.values():
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                if title_parts:
                    return "".join(t.get("plain_text", "") for t in title_parts)

        # Fallback: check child_page block title in parent context
        return "Untitled"

    def _resolve_relation_titles(self, page):
        """Resolve relation property IDs to titles in-place so downstream can render them."""
        properties = page.get("properties", {})
        for prop in properties.values():
            if prop.get("type") != "relation":
                continue
            relations = prop.get("relation", [])
            for rel in relations:
                if "id" in rel and "title" not in rel:
                    try:
                        related_page = self._get_cached_page(rel["id"])
                        if related_page:
                            rel["title"] = self.get_page_title(related_page)
                    except Exception:
                        pass

    def _api_call_with_retry(self, call, max_retries=3):
        """Execute an API call with retry on transient network errors."""
        for attempt in range(max_retries):
            try:
                return call()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    return None

    def delay(self):
        if self.request_delay > 0:
            time.sleep(self.request_delay)
