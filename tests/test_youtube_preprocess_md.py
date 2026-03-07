import os
import tempfile

from youtube_preprocess_md import (
    parse_time_to_seconds,
    parse_frontmatter,
    extract_sections,
    extract_description_summary,
    extract_chapters,
    process_transcript,
    build_output,
    process_file,
)


class TestParseTimeToSeconds:
    def test_mm_ss(self):
        assert parse_time_to_seconds("02:03") == 123

    def test_single_digit_minutes(self):
        assert parse_time_to_seconds("0:00") == 0

    def test_h_mm_ss(self):
        assert parse_time_to_seconds("1:02:03") == 3723

    def test_zero(self):
        assert parse_time_to_seconds("00:00") == 0


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        lines = [
            "---",
            "title: My Title",
            'url: "https://example.com"',
            "video_id: abc123",
            "---",
            "",
            "# My Title",
        ]
        meta, rest = parse_frontmatter(lines)
        assert meta["title"] == "My Title"
        assert meta["url"] == "https://example.com"
        assert meta["video_id"] == "abc123"
        assert rest[0] == ""
        assert rest[1] == "# My Title"

    def test_no_frontmatter(self):
        lines = ["# Just a heading", "Some text"]
        meta, rest = parse_frontmatter(lines)
        assert meta == {}
        assert rest == lines

    def test_unclosed_frontmatter(self):
        lines = ["---", "title: Oops", "no closing delimiter"]
        meta, rest = parse_frontmatter(lines)
        assert meta == {}
        assert rest == lines


class TestExtractSections:
    def test_splits_at_transcript(self):
        body = [
            "# Title",
            "",
            "## Description",
            "Some description.",
            "",
            "## Transcript",
            "[00:00] Hello world.",
            "[00:05] Second line.",
        ]
        desc, trans = extract_sections(body)
        assert "## Transcript" not in desc
        assert len(trans) == 2
        assert "[00:00] Hello world." in trans[0]

    def test_no_transcript_section(self):
        body = ["# Title", "", "Just a body."]
        desc, trans = extract_sections(body)
        assert len(desc) == 3
        assert trans == []


class TestExtractDescriptionSummary:
    def test_extracts_summary_before_emoji(self):
        lines = [
            "# Every Type of Baby Cry Explained",
            "",
            "**Channel**: Emma Hubbard",
            "**Published**: 2025-06-11",
            "**Duration**: 9:19",
            "**Views**: 261,135",
            "**URL**: https://example.com",
            "",
            "## Description",
            "",
            "Your baby's cries are their way of telling you something.",
            "",
            "As always, I hope you find this helpful!",
            "",
            "\U00002705  Get your free guide here: https://example.com",
            "",
            "\U0001F4F1 Follow me on Instagram: https://example.com",
        ]
        summary = extract_description_summary(lines)
        assert "Your baby's cries" in summary
        assert "I hope you find this helpful" in summary
        assert "free guide" not in summary
        assert "Instagram" not in summary

    def test_stops_at_hashtag_line(self):
        lines = [
            "## Description",
            "",
            "Good content here.",
            "",
            "#emmahubbard #babydevelopment",
        ]
        summary = extract_description_summary(lines)
        assert "Good content" in summary
        assert "emmahubbard" not in summary

    def test_stops_at_disclaimer(self):
        lines = [
            "## Description",
            "",
            "Content paragraph.",
            "",
            "Disclaimer:",
            "This is not medical advice.",
        ]
        summary = extract_description_summary(lines)
        assert "Content paragraph" in summary
        assert "medical advice" not in summary

    def test_stops_at_chapter_timestamps(self):
        lines = [
            "## Description",
            "",
            "Summary text.",
            "",
            "00:00 - 01:02 : First Chapter",
            "01:03 - 02:00 : Second Chapter",
        ]
        summary = extract_description_summary(lines)
        assert "Summary text" in summary
        assert "First Chapter" not in summary

    def test_empty_description(self):
        lines = [
            "## Description",
            "",
            "\U00002705 Get your free guide",
        ]
        summary = extract_description_summary(lines)
        assert summary == ""

    def test_starts_with_emoji_promo(self):
        lines = [
            "## Description",
            "",
            "\U00002705 Get your free guide: https://example.com",
            "",
            "Actual content here.",
        ]
        summary = extract_description_summary(lines)
        assert summary == ""


class TestExtractChapters:
    def test_standard_format(self):
        lines = [
            "Some text",
            "00:00 - 00:31 : First Chapter",
            "00:32 - 02:03 : Second Chapter",
        ]
        chapters = extract_chapters(lines)
        assert len(chapters) == 2
        assert chapters[0] == (0, "First Chapter")
        assert chapters[1] == (32, "Second Chapter")

    def test_no_space_before_colon(self):
        lines = ["00:00 - 02:03: Title Here"]
        chapters = extract_chapters(lines)
        assert len(chapters) == 1
        assert chapters[0] == (0, "Title Here")

    def test_dash_separator(self):
        lines = ["00:00 - 02:03 - Title Here"]
        chapters = extract_chapters(lines)
        assert len(chapters) == 1
        assert chapters[0] == (0, "Title Here")

    def test_start_only(self):
        lines = ["00:00 - First Topic"]
        chapters = extract_chapters(lines)
        assert len(chapters) == 1
        assert chapters[0] == (0, "First Topic")

    def test_single_digit_minutes(self):
        lines = ["0:00 - 2:03 : Title"]
        chapters = extract_chapters(lines)
        assert len(chapters) == 1
        assert chapters[0] == (0, "Title")

    def test_no_chapters(self):
        lines = ["Just regular text", "No timestamps here"]
        chapters = extract_chapters(lines)
        assert chapters == []

    def test_sorted_by_start(self):
        lines = [
            "05:00 - 06:00 : Late Chapter",
            "00:00 - 02:00 : Early Chapter",
        ]
        chapters = extract_chapters(lines)
        assert chapters[0][0] < chapters[1][0]


class TestProcessTranscript:
    def test_strips_timestamps_and_merges(self):
        lines = [
            "[00:00] Hello world.",
            "[00:05] This is the second line.",
            "",
            "[00:10] New paragraph here.",
        ]
        result = process_transcript(lines, chapters=[])
        assert "Hello world. This is the second line." in result
        assert "New paragraph here." in result
        assert "[00:00]" not in result

    def test_with_chapters(self):
        chapters = [(0, "Intro"), (10, "Main Content")]
        lines = [
            "[00:00] Welcome to the show.",
            "[00:05] Let me introduce myself.",
            "[00:10] Now for the main content.",
            "[00:15] This is important stuff.",
        ]
        result = process_transcript(lines, chapters)
        assert "## Intro" in result
        assert "## Main Content" in result
        assert "Welcome to the show." in result
        assert "main content." in result

    def test_chapter_breaks_paragraph(self):
        chapters = [(0, "Part 1"), (5, "Part 2")]
        lines = [
            "[00:00] First part content.",
            "[00:05] Second part content.",
        ]
        result = process_transcript(lines, chapters)
        parts = result.split("## Part 2")
        assert "First part content." in parts[0]
        assert "Second part content." in parts[1]

    def test_continuation_lines(self):
        lines = [
            "[00:00] First line",
            "continues here.",
            "",
            "[00:05] Next paragraph.",
        ]
        result = process_transcript(lines, chapters=[])
        assert "First line continues here." in result

    def test_dash_after_timestamp(self):
        lines = [
            "[00:00] - Toddlers have a genuine",
            "fear of trying new food.",
        ]
        result = process_transcript(lines, chapters=[])
        assert "Toddlers have a genuine fear of trying new food." in result
        assert "[00:00]" not in result


class TestBuildOutput:
    def test_basic_output(self):
        metadata = {"title": "Test Title", "url": "https://example.com", "channel": "TestChan"}
        result = build_output(
            title="Test Title",
            metadata=metadata,
            summary="This is a summary.",
            transcript_text="Content here.",
        )
        assert "---" in result
        assert "title: Test Title" in result
        assert 'url: "https://example.com"' in result
        assert "channel: TestChan" in result
        assert "# Test Title" in result
        assert "This is a summary." in result
        assert "Content here." in result

    def test_no_summary(self):
        metadata = {"title": "Title", "url": "https://example.com"}
        result = build_output(
            title="Title",
            metadata=metadata,
            summary="",
            transcript_text="Content.",
        )
        assert "# Title" in result
        assert "Content." in result
        # Should not have double blank lines between title and content
        assert "# Title\n\nContent." in result

    def test_with_chapters_in_transcript(self):
        metadata = {"title": "Title"}
        result = build_output(
            title="Title",
            metadata=metadata,
            summary="Summary.",
            transcript_text="## Chapter 1\n\nContent.",
        )
        assert "## Chapter 1" in result

    def test_preserves_keep_frontmatter_fields(self):
        metadata = {
            "title": "Vid",
            "url": "https://example.com",
            "video_id": "abc123",
            "channel": "MyChan",
            "upload_date": "2025-01-01",
            "duration": "120",
            "view_count": "1000",
        }
        result = build_output(title="Vid", metadata=metadata, summary="", transcript_text="Text.")
        # Kept fields
        assert "title: Vid" in result
        assert 'url: "https://example.com"' in result
        assert "video_id: abc123" in result
        assert "channel: MyChan" in result
        assert "upload_date: 2025-01-01" in result
        # Dropped fields
        assert "duration" not in result
        assert "view_count" not in result


class TestProcessFile:
    def test_full_file_with_chapters(self):
        content = """---
title: Test Video
video_id: abc123
channel: Test Channel
url: "https://www.youtube.com/watch?v=abc123"
upload_date: 2025-01-01
duration: 120
view_count: 1000
description: "Some description text"
fetched_at: "2025-01-01T00:00:00+00:00"
---

# Test Video

**Channel**: Test Channel
**Published**: 2025-01-01
**Duration**: 2:00
**Views**: 1,000
**URL**: https://www.youtube.com/watch?v=abc123

## Description

This video explains something important.

As always, I hope you find this helpful!

\u2705 Get your free guide: https://example.com

\U0001F4F1 Follow me: https://example.com

#test #video

00:00 - 00:30 : Introduction
00:31 - 01:00 : Main Topic
01:01 - 02:00 : Conclusion

Disclaimer:
Not medical advice.

## Transcript

[00:00] Welcome to this video.
[00:05] I'm going to explain something.
[00:31] Now for the main topic.
[00:40] This is really important.
[01:01] That's all for today.
[01:10] Thanks for watching."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            tmp_path = f.name

        try:
            output, stats = process_file(tmp_path)
            assert output is not None
            assert stats["chapters"] == 3
            assert stats["has_summary"] is True

            # Check frontmatter preserves KEEP_FRONTMATTER fields only
            frontmatter = output.split("---")[1]
            assert "title: Test Video" in frontmatter
            assert "video_id: abc123" in frontmatter
            assert "channel: Test Channel" in frontmatter
            assert "upload_date: 2025-01-01" in frontmatter
            assert "duration" not in frontmatter
            assert "view_count" not in frontmatter
            assert "fetched_at" not in frontmatter

            # Check description summary
            assert "explains something important" in output
            assert "free guide" not in output
            assert "Disclaimer" not in output

            # Check chapter headings
            assert "## Introduction" in output
            assert "## Main Topic" in output
            assert "## Conclusion" in output

            # Check transcript content
            assert "Welcome to this video." in output
            assert "[00:00]" not in output

            # Check noise is removed
            assert "**Channel**" not in output
            assert "#test" not in output
        finally:
            os.unlink(tmp_path)

    def test_full_file_without_chapters(self):
        content = """---
title: Simple Video
video_id: xyz789
url: "https://www.youtube.com/watch?v=xyz789"
---

# Simple Video

**Channel**: Test
**Published**: 2025-01-01
**Duration**: 1:00
**Views**: 500
**URL**: https://www.youtube.com/watch?v=xyz789

## Description

A short description.

## Transcript

[00:00] Just a simple video.
[00:05] With no chapters at all.

[00:10] And a second paragraph."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            tmp_path = f.name

        try:
            output, stats = process_file(tmp_path)
            assert output is not None
            assert stats["chapters"] == 0
            assert "##" not in output.split("# Simple Video")[1]  # No H2 headings
            assert "Just a simple video. With no chapters at all." in output
            assert "And a second paragraph." in output
        finally:
            os.unlink(tmp_path)

    def test_no_title_returns_none(self):
        content = """---
video_id: notitle
---

Some content."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            tmp_path = f.name

        try:
            output, stats = process_file(tmp_path)
            assert output is None
        finally:
            os.unlink(tmp_path)


class TestMainCli:
    def test_dry_run(self):
        with tempfile.TemporaryDirectory() as input_dir:
            # Create a test file
            content = """---
title: CLI Test
url: "https://example.com"
---

# CLI Test

## Description

Test content.

## Transcript

[00:00] Hello."""
            with open(os.path.join(input_dir, "test.md"), "w") as f:
                f.write(content)

            output_dir = os.path.join(input_dir, "output")

            from youtube_preprocess_md import main
            import sys

            old_argv = sys.argv
            sys.argv = [
                "youtube_preprocess_md.py",
                "--inputDir", input_dir,
                "--outputDir", output_dir,
                "--dryRun",
            ]
            try:
                main()
                # Output dir should not be created in dry run
                assert not os.path.exists(output_dir)
            finally:
                sys.argv = old_argv

    def test_actual_write(self):
        with tempfile.TemporaryDirectory() as input_dir:
            content = """---
title: Write Test
url: "https://example.com"
---

# Write Test

## Description

Summary here.

## Transcript

[00:00] Content here."""
            with open(os.path.join(input_dir, "test.md"), "w") as f:
                f.write(content)

            output_dir = os.path.join(input_dir, "output")

            from youtube_preprocess_md import main
            import sys

            old_argv = sys.argv
            sys.argv = [
                "youtube_preprocess_md.py",
                "--inputDir", input_dir,
                "--outputDir", output_dir,
            ]
            try:
                main()
                out_file = os.path.join(output_dir, "test.md")
                assert os.path.exists(out_file)
                with open(out_file) as f:
                    result = f.read()
                assert "# Write Test" in result
                assert "Summary here." in result
                assert "Content here." in result
            finally:
                sys.argv = old_argv
