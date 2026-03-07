from main.sources.files.markdown_heading_splitter import MarkdownHeadingSplitter


def make_splitter(chunk_size=1000, chunk_overlap=100):
    return MarkdownHeadingSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class TestNoHeadingsFallback:
    def test_no_headings_falls_back_to_plain_split(self):
        splitter = make_splitter()
        text = "This is plain text with no headings at all."
        chunks = splitter.split(text)
        assert len(chunks) >= 1
        assert all(c["heading"] is None for c in chunks)
        assert "plain text" in chunks[0]["text"]

    def test_no_headings_large_text_produces_multiple_chunks(self):
        splitter = make_splitter(chunk_size=50, chunk_overlap=0)
        text = "Word " * 100  # ~500 chars
        chunks = splitter.split(text)
        assert len(chunks) > 1
        assert all(c["heading"] is None for c in chunks)


class TestSingleHeading:
    def test_single_heading_with_body(self):
        splitter = make_splitter()
        text = "# Introduction\nThis is the introduction section."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Introduction"
        assert "introduction section" in chunks[0]["text"]

    def test_heading_line_not_in_body(self):
        splitter = make_splitter()
        text = "# My Heading\nBody text here."
        chunks = splitter.split(text)
        assert "# My Heading" not in chunks[0]["text"]


class TestMultipleHeadings:
    def test_multiple_headings_produce_multiple_chunks(self):
        splitter = make_splitter()
        text = "# First\nContent one.\n\n## Second\nContent two.\n\n### Third\nContent three."
        chunks = splitter.split(text)
        assert len(chunks) == 3
        assert chunks[0]["heading"] == "First"
        assert chunks[1]["heading"] == "Second"
        assert chunks[2]["heading"] == "Third"
        assert "Content one" in chunks[0]["text"]
        assert "Content two" in chunks[1]["text"]
        assert "Content three" in chunks[2]["text"]


class TestPreamble:
    def test_preamble_before_first_heading(self):
        splitter = make_splitter()
        text = "Some intro text before headings.\n\n# Section One\nSection content."
        chunks = splitter.split(text)
        assert len(chunks) == 2
        assert chunks[0]["heading"] is None
        assert "intro text" in chunks[0]["text"]
        assert chunks[1]["heading"] == "Section One"

    def test_whitespace_only_preamble_skipped(self):
        splitter = make_splitter()
        text = "   \n\n# Section One\nContent here."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Section One"

    def test_newline_only_preamble_skipped(self):
        splitter = make_splitter()
        text = "\n# Section\nBody."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Section"


class TestMixedHeadingLevels:
    def test_h1_h2_h3_each_start_new_section(self):
        splitter = make_splitter()
        text = "# H1\nH1 body.\n\n## H2\nH2 body.\n\n### H3\nH3 body."
        chunks = splitter.split(text)
        assert len(chunks) == 3
        assert chunks[0]["heading"] == "H1"
        assert chunks[1]["heading"] == "H2"
        assert chunks[2]["heading"] == "H3"


class TestH4PlusIgnored:
    def test_h4_not_treated_as_section_boundary(self):
        splitter = make_splitter()
        text = "# Main Section\nIntro.\n\n#### Detail\nThis stays in the main section."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Main Section"
        assert "#### Detail" in chunks[0]["text"]
        assert "stays in the main section" in chunks[0]["text"]

    def test_h5_h6_not_treated_as_boundaries(self):
        splitter = make_splitter()
        text = "# Top\nBody.\n\n##### Deep heading\nDeep content.\n\n###### Deepest\nMore."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Top"


class TestLargeSectionSubSplit:
    def test_large_section_sub_split_shares_heading(self):
        splitter = make_splitter(chunk_size=50, chunk_overlap=0)
        body = "This is a sentence. " * 20  # ~400 chars
        text = f"## Big Section\n{body}"
        chunks = splitter.split(text)
        assert len(chunks) > 1
        assert all(c["heading"] == "Big Section" for c in chunks)

    def test_sub_chunks_cover_full_content(self):
        splitter = make_splitter(chunk_size=50, chunk_overlap=0)
        body = "Word " * 50
        text = f"# Section\n{body}"
        chunks = splitter.split(text)
        combined = " ".join(c["text"] for c in chunks)
        assert "Word" in combined


class TestATXClosingSequence:
    def test_trailing_hashes_stripped(self):
        splitter = make_splitter()
        text = "# Heading ##\nBody text."
        chunks = splitter.split(text)
        assert chunks[0]["heading"] == "Heading"

    def test_trailing_hashes_with_spaces_stripped(self):
        splitter = make_splitter()
        text = "## Section ###  \nContent."
        chunks = splitter.split(text)
        assert chunks[0]["heading"] == "Section"

    def test_hashes_without_preceding_space_kept(self):
        splitter = make_splitter()
        text = "# Heading#tag\nBody."
        chunks = splitter.split(text)
        assert chunks[0]["heading"] == "Heading#tag"


class TestUnicodeAndSpecialChars:
    def test_norwegian_characters_in_heading(self):
        splitter = make_splitter()
        text = "# Lønn og fravær\nInformasjon om lønn."
        chunks = splitter.split(text)
        assert chunks[0]["heading"] == "Lønn og fravær"
        assert "lønn" in chunks[0]["text"]

    def test_emoji_in_heading(self):
        splitter = make_splitter()
        text = "# \U0001f680 Getting Started\nWelcome aboard."
        chunks = splitter.split(text)
        assert chunks[0]["heading"] == "\U0001f680 Getting Started"

    def test_inline_markdown_formatting_in_heading(self):
        splitter = make_splitter()
        text = "# **Bold** and *italic* heading\nBody content."
        chunks = splitter.split(text)
        assert chunks[0]["heading"] == "**Bold** and *italic* heading"


class TestEmptySections:
    def test_empty_section_between_headings_skipped(self):
        splitter = make_splitter()
        text = "# First\n\n## Second\nSecond content."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Second"

    def test_all_empty_sections_produces_no_chunks(self):
        splitter = make_splitter()
        text = "# A\n\n## B\n\n### C\n"
        chunks = splitter.split(text)
        assert len(chunks) == 0
