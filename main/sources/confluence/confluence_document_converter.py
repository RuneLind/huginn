import os

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ConfluenceDocumentConverter:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

    def _get_page(self, document):
        """Extract the page object from a raw document. Override for Cloud format."""
        return document["page"]

    def convert(self, document):
        page = self._get_page(document)
        return [{
            "id": page['id'],
            "url": self._build_url(page),
            "modifiedTime": page['version']['when'],
            "text": self._build_document_text(page, document["comments"]),
            "chunks": self._split_to_chunks(page, document["comments"])
        }]

    def _build_document_text(self, page, comments):
        title = self._build_path_of_titles(page)
        body_and_comments = self._fetch_body_and_comments(page, comments)

        return self._convert_to_text([title, body_and_comments])

    def _split_to_chunks(self, page, comments):
        chunks = [{
                "indexedData": self._build_path_of_titles(page),
            }]

        body_and_comments = self._fetch_body_and_comments(page, comments)

        if body_and_comments:
            for chunk in self.text_splitter.split_text(body_and_comments):
                chunks.append({
                    "indexedData": chunk
                })

        return chunks

    def _fetch_body_and_comments(self, page, comments):
        body = self._get_cleaned_body(page)
        comment_texts = [self._get_cleaned_body(comment) for comment in comments]

        return self._convert_to_text([body] + comment_texts)

    def _convert_to_text(self, elements, delimiter="\n\n"):
        return delimiter.join([element for element in elements if element])

    def _get_cleaned_body(self, document):
        document_text_html = document["body"]["storage"]["value"]
        if not document_text_html:
            return ""

        soup = BeautifulSoup(document_text_html, "html.parser")
        return soup.get_text(separator=os.linesep, strip=True)

    def _build_path_of_titles(self, document):
        page_title = [document['title']] if 'title' in document else []
        return " -> ".join([ ancestor["title"] for ancestor in document['ancestors'] if "title" in ancestor ] + page_title)

    def _build_url(self, page):
        base_url = page['_links']['self'].split("/rest/api/")[0]
        return f"{base_url}{page['_links']['webui']}"
