PREFIX_SYSTEM_PROMPT = """You are writing short retrieval-anchoring prefixes for document chunks.

For each chunk, write a 50–100 token prefix that answers:
- Which document is this chunk from? (Title, section if knowable.)
- What is the chunk about, in the language of the surrounding document?
- What domain terms does it assume the reader knows?

Write in the same language as the document. For Norwegian documents, write the prefix in Norwegian.

Be concrete: name the page, the system, and the concept. Do not write generic openers like "This chunk discusses important concepts" — name the thing.

Return a JSON array of strings, one prefix per chunk, in the same order as the chunks. No commentary, no markdown, just the JSON array."""


PREFIX_USER_PROMPT_TEMPLATE = """<document>
{document_text}
</document>

<chunks>
{numbered_chunks}
</chunks>

Return a JSON array of {chunk_count} prefixes, in order."""


def format_numbered_chunks(chunks: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}]\n{chunk}" for i, chunk in enumerate(chunks))


def render_user_prompt(document_text: str, chunks: list[str]) -> str:
    return PREFIX_USER_PROMPT_TEMPLATE.format(
        document_text=document_text,
        numbered_chunks=format_numbered_chunks(chunks),
        chunk_count=len(chunks),
    )
