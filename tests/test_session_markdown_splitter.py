from main.sources.files.session_markdown_splitter import SessionMarkdownSplitter


def make_splitter(target_chars=2500, min_chars=400):
    return SessionMarkdownSplitter(target_chars=target_chars, min_chars=min_chars)


SAMPLE_SESSION = """---
session_id: "abc123"
project: "my-project"
---

# Sample session

## User

How do I set up authentication in this project?

## Assistant

<details><summary>Thinking</summary>

Let me look at the auth setup...

</details>

## Assistant

- [Tool: Read] /src/auth.py

## Assistant

The project uses JWT authentication. Here's how it works:

1. Login endpoint at `/api/login` accepts username/password
2. Returns a JWT token valid for 24 hours
3. All protected routes check the `Authorization` header

You can configure the secret key in `settings.py`.

## User

Can you add a refresh token mechanism?

## Assistant

<details><summary>Thinking</summary>

The user wants refresh tokens. I'll need to modify the auth module.

</details>

## Assistant

- [Tool: Read] /src/settings.py

## Assistant

Sure! I'll add a refresh token mechanism. Here's the approach:

1. Generate a long-lived refresh token alongside the JWT
2. Store refresh tokens in the database with expiry
3. Add a `/api/refresh` endpoint

Let me implement this now.

## Assistant

- [Tool: Edit] /src/auth.py

## Assistant

I've added the refresh token mechanism:

- New `RefreshToken` model in the database
- `/api/refresh` endpoint that validates and rotates refresh tokens
- Refresh tokens expire after 30 days
- Old refresh tokens are invalidated on rotation

## User

What about token revocation?

## Assistant

Good question! For token revocation, we have a few options:

1. **Blocklist approach**: Store revoked tokens in Redis with TTL matching token expiry
2. **Short-lived tokens**: Keep JWT lifetime very short (5 min) and rely on refresh tokens
3. **Token versioning**: Add a version field to user model, increment on revocation

I recommend option 2 (short-lived + refresh) since we already have refresh tokens. The JWT can be 5 minutes, and refresh token rotation handles the rest.
"""


class TestBasicSplitting:
    def test_produces_chunks(self):
        splitter = make_splitter()
        chunks = splitter.split(SAMPLE_SESSION)
        assert len(chunks) >= 1
        assert all("text" in c and "heading" in c for c in chunks)

    def test_chunks_have_turn_headings(self):
        splitter = make_splitter()
        chunks = splitter.split(SAMPLE_SESSION)
        for chunk in chunks:
            assert chunk["heading"] is not None
            assert "Turn" in chunk["heading"]

    def test_reduces_chunk_count_vs_heading_splitter(self):
        """Session splitter should produce far fewer chunks than heading-based splitting."""
        from main.sources.files.markdown_heading_splitter import MarkdownHeadingSplitter
        heading_splitter = MarkdownHeadingSplitter(chunk_size=1000, chunk_overlap=100)
        session_splitter = make_splitter()

        heading_chunks = heading_splitter.split(SAMPLE_SESSION)
        session_chunks = session_splitter.split(SAMPLE_SESSION)

        # Session splitter should produce fewer, denser chunks
        assert len(session_chunks) < len(heading_chunks)


class TestNoiseStripping:
    def test_thinking_blocks_removed(self):
        splitter = make_splitter()
        chunks = splitter.split(SAMPLE_SESSION)
        full_text = " ".join(c["text"] for c in chunks)
        assert "<details>" not in full_text
        assert "Thinking" not in full_text
        assert "Let me look at the auth setup" not in full_text

    def test_tool_lines_removed(self):
        splitter = make_splitter()
        chunks = splitter.split(SAMPLE_SESSION)
        full_text = " ".join(c["text"] for c in chunks)
        assert "[Tool:" not in full_text
        assert "/src/auth.py" not in full_text

    def test_substantive_content_preserved(self):
        splitter = make_splitter()
        chunks = splitter.split(SAMPLE_SESSION)
        full_text = " ".join(c["text"] for c in chunks)
        assert "JWT authentication" in full_text
        assert "refresh token" in full_text
        assert "token revocation" in full_text


class TestExchangeGrouping:
    def test_user_and_assistant_paired(self):
        splitter = make_splitter()
        chunks = splitter.split(SAMPLE_SESSION)
        for chunk in chunks:
            assert "**User:**" in chunk["text"]
            assert "**Assistant:**" in chunk["text"]

    def test_multiple_assistant_turns_merged(self):
        """Multiple consecutive assistant turns should be merged into one exchange."""
        splitter = make_splitter()
        text = """## User

Question here.

## Assistant

- [Tool: Read] /file.py

## Assistant

Here is my answer about the file.

## User

Follow-up question.

## Assistant

Follow-up answer.
"""
        chunks = splitter.split(text)
        # Should produce 2 exchanges (2 user turns), possibly in 1-2 chunks
        assert len(chunks) >= 1
        # First exchange should contain the substantive answer, not tool line
        assert "my answer about the file" in chunks[0]["text"]


class TestSmallSession:
    def test_single_exchange(self):
        splitter = make_splitter()
        text = """## User

Hello, how are you?

## Assistant

I'm doing well! How can I help you today?
"""
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Turn 1"
        assert "Hello" in chunks[0]["text"]
        assert "doing well" in chunks[0]["text"]


class TestNoTurns:
    def test_no_turn_headings_returns_cleaned_text(self):
        splitter = make_splitter(min_chars=0)
        text = "Just some plain text without any turn markers."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["heading"] is None
        assert "plain text" in chunks[0]["text"]

    def test_empty_text_returns_empty(self):
        splitter = make_splitter()
        chunks = splitter.split("")
        assert len(chunks) == 0


class TestOverlap:
    def test_overlap_between_chunks(self):
        """When there are enough exchanges to produce multiple chunks, they should overlap."""
        splitter = make_splitter(target_chars=200, min_chars=50)
        exchanges = []
        for i in range(6):
            exchanges.append(f"## User\n\nQuestion {i}: " + "x " * 30 + f"\n\n## Assistant\n\nAnswer {i}: " + "y " * 30)
        text = "\n\n".join(exchanges)

        chunks = splitter.split(text)
        if len(chunks) >= 2:
            # Check overlap: last exchange of chunk N should appear in chunk N+1
            # We can verify by checking that some content appears in consecutive chunks
            for i in range(len(chunks) - 1):
                # With 1-exchange overlap, the last exchange text in chunk i
                # should appear at the start of chunk i+1
                chunk_a_lines = chunks[i]["text"]
                chunk_b_lines = chunks[i + 1]["text"]
                # There should be some shared content
                # (we can't be exact since formatting may vary, but the overlap exchange should be in both)
                assert len(set(chunk_a_lines.split()) & set(chunk_b_lines.split())) > 0


    def test_no_duplicate_final_chunk(self):
        """Overlap should not produce a trailing chunk that's pure repetition of the previous chunk's ending."""
        splitter = make_splitter(target_chars=200, min_chars=0)
        exchanges = []
        for i in range(4):
            exchanges.append(f"## User\n\nQuestion {i}: " + "x " * 30 + f"\n\n## Assistant\n\nAnswer {i}: " + "y " * 30)
        text = "\n\n".join(exchanges)
        chunks = splitter.split(text)

        if len(chunks) >= 2:
            # The last chunk should contain content NOT in the second-to-last chunk
            last_text = chunks[-1]["text"]
            prev_text = chunks[-2]["text"]
            # Last chunk should not be a pure subset of the previous chunk
            last_words = set(last_text.split())
            prev_words = set(prev_text.split())
            new_words = last_words - prev_words
            assert len(new_words) > 0, "Last chunk is pure repetition of previous chunk"


class TestLeadingAssistantTurns:
    def test_leading_assistant_turns_ignored(self):
        """Assistant turns before any user turn should be silently dropped."""
        splitter = make_splitter()
        text = """## Assistant

Some preamble from the assistant before any user message.

## User

Hello, what can you do?

## Assistant

I can help with code and questions!
"""
        chunks = splitter.split(text)
        assert len(chunks) == 1
        full_text = chunks[0]["text"]
        assert "preamble" not in full_text
        assert "Hello" in full_text
        assert "help with code" in full_text


class TestMinCharsFilter:
    def test_short_noise_chunks_filtered(self):
        """Chunks below min_chars should be skipped (except the first chunk)."""
        splitter = make_splitter(target_chars=200, min_chars=100)
        text = """## User

Short question.

## Assistant

- [Tool: Read] /file.py

## User

Another longer question that has more substance and detail about the topic at hand which makes it more meaningful.

## Assistant

A detailed answer that provides real value and explains the concept thoroughly with examples and code references.
"""
        chunks = splitter.split(text)
        # All chunks should exist (at least one), and the short noise-only one may be filtered
        assert len(chunks) >= 1
