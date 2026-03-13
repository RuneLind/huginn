from .confluence_document_converter import ConfluenceDocumentConverter


class ConfluenceCloudDocumentConverter(ConfluenceDocumentConverter):
    """Confluence Cloud variant — pages are nested under document["page"]["content"]."""

    def _get_page(self, document):
        return document["page"]["content"]
