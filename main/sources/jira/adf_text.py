"""Flatten Atlassian Document Format (ADF) rich-text fields to plain text.

Kept dependency-free on purpose: both the indexing converter
(JiraCloudDocumentConverter) and the standalone Playwright fetcher
(scripts/jira/fetchers/jira_fetcher.py) import this, and the fetcher must not
pull in the indexing layer's heavier dependencies (e.g. langchain).
"""

# ADF block nodes that should end a line, so list items / table cells /
# paragraphs don't run together when the node tree is flattened.
ADF_BLOCK_TYPES = frozenset({
    "paragraph", "heading", "listItem", "blockquote", "panel",
    "tableRow", "tableCell", "tableHeader", "codeBlock", "rule",
})


def adf_to_text(node_tree) -> str:
    """Extract plain text from an ADF node tree (a dict).

    Walks recursively so text nested in bullet/numbered lists, tables, panels
    and blockquotes is captured, with block nodes terminating a line.
    """
    if not isinstance(node_tree, dict):
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
            if node.get("type") in ADF_BLOCK_TYPES:
                parts.append("\n")

    walk(node_tree)
    return "\n".join(line.strip() for line in "".join(parts).splitlines() if line.strip()).strip()
