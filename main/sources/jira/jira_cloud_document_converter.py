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

    # ADF block nodes that should end a line, so list items / table cells /
    # paragraphs don't run together when flattened.
    _ADF_BLOCK_TYPES = frozenset({
        "paragraph", "heading", "listItem", "blockquote", "panel",
        "tableRow", "tableCell", "tableHeader", "codeBlock", "rule",
    })

    @classmethod
    def _convert_adf_to_text(cls, field_with_content):
        """Extract plain text from an Atlassian Document Format (ADF) field.

        Walks the node tree recursively so text nested more than two levels deep
        — bullet and numbered lists, tables, panels, blockquotes — is captured,
        not just the two-level-deep paragraphs the previous version reached.
        """
        if not isinstance(field_with_content, dict):
            return ""

        parts = []

        def walk(node):
            if not isinstance(node, dict):
                return
            text = node.get("text")
            if text:
                parts.append(text)
            if node.get("type") == "hardBreak":
                parts.append("\n")
            children = node.get("content")
            if isinstance(children, list):
                for child in children:
                    walk(child)
                if node.get("type") in cls._ADF_BLOCK_TYPES:
                    parts.append("\n")

        walk(field_with_content)
        return "\n".join(line.strip() for line in "".join(parts).splitlines() if line.strip()).strip()
