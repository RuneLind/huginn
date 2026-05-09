from main.utils.frontmatter import (
    read_frontmatter,
    read_frontmatter_from_path,
    strip_frontmatter,
)


class TestReadFrontmatter:
    def test_no_frontmatter_returns_empty(self):
        assert read_frontmatter("just body text") == {}

    def test_empty_string(self):
        assert read_frontmatter("") == {}

    def test_basic_key_value(self):
        text = '---\ntitle: foo\nstatus: open\n---\nbody'
        assert read_frontmatter(text) == {"title": "foo", "status": "open"}

    def test_strips_outer_double_quotes(self):
        text = '---\ntitle: "foo bar"\n---\n'
        assert read_frontmatter(text) == {"title": "foo bar"}

    def test_drops_empty_values_unless_followed_by_list(self):
        """Empty value with no list items is silently dropped."""
        text = '---\ntitle: foo\nempty:\n---\n'
        assert read_frontmatter(text) == {"title": "foo"}

    def test_yaml_list_joined_with_commas(self):
        """`key:` with empty value followed by `- item` lines yields a comma-joined string —
        downstream code expects str values, not real lists."""
        text = '---\nlabels:\n  - bug\n  - urgent\nstatus: open\n---\n'
        assert read_frontmatter(text) == {"labels": "bug,urgent", "status": "open"}

    def test_list_resets_when_new_key_seen(self):
        text = '---\nlabels:\n  - bug\nstatus: open\n  - not_a_list_item\n---\n'
        result = read_frontmatter(text)
        assert result["labels"] == "bug"
        assert result["status"] == "open"

    def test_handles_tolerant_dashes(self):
        """Trailing whitespace on the dash lines is tolerated."""
        assert read_frontmatter("---  \ntitle: foo\n---  \nbody") == {"title": "foo"}

    def test_no_frontmatter_when_dashes_not_at_start(self):
        text = 'first line\n---\ntitle: foo\n---\n'
        assert read_frontmatter(text) == {}

    def test_value_with_colon(self):
        """A colon inside the value is preserved (only the first colon splits)."""
        text = '---\nurl: https://example.com/path\n---\n'
        assert read_frontmatter(text) == {"url": "https://example.com/path"}


class TestReadFrontmatterFromPath:
    def test_reads_file(self, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text('---\ntitle: foo\n---\nbody')
        assert read_frontmatter_from_path(str(p)) == {"title": "foo"}

    def test_missing_file_returns_empty(self, tmp_path):
        assert read_frontmatter_from_path(str(tmp_path / "missing.md")) == {}

    def test_reads_only_head(self, tmp_path):
        """Frontmatter readers only need to scan the first ~8KB."""
        p = tmp_path / "big.md"
        body = "x" * 50000
        p.write_text(f'---\ntitle: foo\n---\n{body}')
        assert read_frontmatter_from_path(str(p)) == {"title": "foo"}


class TestStripFrontmatter:
    def test_removes_block_and_trailing_newline(self):
        text = '---\ntitle: foo\n---\nbody line\n'
        assert strip_frontmatter(text) == 'body line\n'

    def test_no_frontmatter_returns_unchanged(self):
        assert strip_frontmatter('plain body') == 'plain body'

    def test_only_strips_leading_block(self):
        text = '---\ntitle: foo\n---\nbody\n---\nnot frontmatter\n---\n'
        assert strip_frontmatter(text) == 'body\n---\nnot frontmatter\n---\n'
