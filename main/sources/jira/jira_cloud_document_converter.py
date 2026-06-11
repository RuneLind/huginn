from .jira_document_converter import JiraDocumentConverter
from .adf_text import adf_to_text


class JiraCloudDocumentConverter(JiraDocumentConverter):
    """Jira Cloud variant — description and comments use ADF (Atlassian Document Format)."""

    def _get_description_text(self, document):
        description = document['fields']['description']
        if not description:
            return ""
        return adf_to_text(description)

    def _get_comment_text(self, comment):
        return adf_to_text(comment['body'])

    # The ADF flattener now lives in adf_text.py (shared with the fetcher).
    # Kept as an alias for backward compatibility / existing tests.
    _convert_adf_to_text = staticmethod(adf_to_text)
