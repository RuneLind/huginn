from .jira_document_converter import JiraDocumentConverter


class JiraCloudDocumentConverter(JiraDocumentConverter):
    """Jira Cloud variant — description and comments use ADF (Atlassian Document Format)."""

    def _get_description_text(self, document):
        description = document['fields']['description']
        if not description:
            return ""
        return self._convert_adf_to_text(description)

    def _get_comment_text(self, comment):
        return self._convert_adf_to_text(comment['body'])

    @staticmethod
    def _convert_adf_to_text(field_with_content):
        """Extract plain text from an Atlassian Document Format (ADF) field."""
        texts = []
        for content in field_with_content.get("content", []):
            if "content" in content:
                for content_of_content in content["content"]:
                    if "text" in content_of_content:
                        texts.append(content_of_content["text"])
        return "\n".join([t for t in texts if t]).strip()
