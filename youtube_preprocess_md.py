"""
Preprocess YouTube transcript markdown files for better chunking.

Strips noise (promo links, disclaimers, metadata) from description,
extracts chapter timestamps to create heading structure, and merges
transcript lines into flowing paragraphs.

Usage:
    uv run youtube_preprocess_md.py --inputDir ./data/sources/youtube-transcripts/markdown/EmmaHubbard \
        --outputDir ./data/sources/youtube-transcripts/processed/EmmaHubbard
    uv run youtube_preprocess_md.py --inputDir ... --outputDir ... --dryRun  # preview
"""

import os
import re
import argparse
import logging

from main.utils.logger import setup_root_logger

setup_root_logger()

# Emoji characters that signal promo/noise lines in descriptions
PROMO_EMOJI_PATTERN = re.compile(
    r"^[\s]*["
    r"\U0001F4F1"  # 📱
    r"\U00002615"  # ☕
    r"\U0001F4D4"  # 📔
    r"\U0000270F"  # ✏
    r"\U00002705"  # ✅
    r"\U0001F476"  # 👶
    r"\U0001F4A1"  # 💡
    r"\U0001F634"  # 😴
    r"\U0001F60A"  # 😊
    r"\U0001F9D1"  # 🧑
    r"\U0001F3AC"  # 🎬
    r"\U0001F4E2"  # 📢
    r"\U0001F381"  # 🎁
    r"\U0001F4DA"  # 📚
    r"\U0001F31F"  # 🌟
    r"\U00002B50"  # ⭐
    r"\U0001F44D"  # 👍
    r"\U0001F64F"  # 🙏
    r"\U0001F4AC"  # 💬
    r"\U0001F4E7"  # 📧
    r"\U0001F517"  # 🔗
    r"\U0001F3AF"  # 🎯
    r"]"
)

# Chapter timestamp formats:
#   00:00 - 02:03 : Title
#   00:00 - 02:03: Title
#   00:00 - 02:03 - Title
#   00:00 - Title (start-only)
#   0:00 - 02:03 : Title
CHAPTER_PATTERN = re.compile(
    r"^(\d{1,2}:\d{2})"          # start time
    r"\s*-\s*"                    # separator
    r"(?:(\d{1,2}:\d{2})\s*[-:]\s*)?"  # optional end time + separator
    r"(.+)$"                      # title
)

# Transcript line timestamp: [MM:SS] or [H:MM:SS]
TRANSCRIPT_TS_PATTERN = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*-?\s*")

# Frontmatter fields to keep
KEEP_FRONTMATTER = {"title", "url", "video_id", "channel", "upload_date"}


def parse_time_to_seconds(time_str):
    """Convert MM:SS or H:MM:SS to total seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def parse_frontmatter(lines):
    """Parse YAML frontmatter, returning (metadata_dict, rest_of_lines)."""
    metadata = {}
    rest_start = 0

    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                rest_start = i + 1
                break
            if ":" in lines[i]:
                key, _, value = lines[i].partition(":")
                metadata[key.strip()] = value.strip().strip('"')
        else:
            # No closing ---, treat everything as body
            rest_start = 0
            metadata = {}

    return metadata, lines[rest_start:]


def extract_sections(body_lines):
    """Split body into description lines and transcript lines."""
    description_lines = []
    transcript_lines = []
    in_transcript = False

    for line in body_lines:
        if line.strip() == "## Transcript":
            in_transcript = True
            continue
        if in_transcript:
            transcript_lines.append(line)
        else:
            description_lines.append(line)

    return description_lines, transcript_lines


def extract_description_summary(description_lines):
    """Extract meaningful summary paragraphs before promo/noise lines.

    Skips the metadata block (Channel, Published, Duration, etc.) and
    the ## Description heading, then takes paragraphs until hitting
    an emoji-prefixed promo line, hashtag line, or Disclaimer.
    """
    summary_lines = []
    started = False

    for line in description_lines:
        stripped = line.strip()

        # Skip metadata block lines
        if stripped.startswith("# "):
            continue
        if stripped.startswith("**") and any(
            stripped.startswith(f"**{k}") for k in ["Channel", "Published", "Duration", "Views", "URL"]
        ):
            continue
        if stripped == "## Description":
            started = True
            continue

        if not started:
            continue

        # Stop at promo emoji lines
        if PROMO_EMOJI_PATTERN.match(stripped):
            break

        # Stop at hashtag lines
        if stripped.startswith("#") and not stripped.startswith("# "):
            break

        # Stop at disclaimer
        if stripped.lower().startswith("disclaimer"):
            break

        # Stop at chapter timestamps (they'll be extracted separately)
        if CHAPTER_PATTERN.match(stripped):
            break

        summary_lines.append(line)

    # Trim trailing blank lines
    while summary_lines and not summary_lines[-1].strip():
        summary_lines.pop()

    return "\n".join(summary_lines).strip()


def extract_chapters(description_lines):
    """Extract chapter timestamps from description.

    Returns list of (start_seconds, title) tuples sorted by start time.
    """
    chapters = []
    for line in description_lines:
        m = CHAPTER_PATTERN.match(line.strip())
        if m:
            start_time = m.group(1)
            title = m.group(3).strip()
            seconds = parse_time_to_seconds(start_time)
            chapters.append((seconds, title))

    chapters.sort(key=lambda c: c[0])
    return chapters


def process_transcript(transcript_lines, chapters):
    """Process transcript lines into structured text.

    Strips timestamps, merges consecutive lines into paragraphs.
    If chapters exist, inserts ## headings at chapter boundaries.
    """
    # Parse transcript into (seconds, text) tuples
    parsed = []
    for line in transcript_lines:
        stripped = line.strip()
        if not stripped:
            parsed.append((None, ""))
            continue

        m = TRANSCRIPT_TS_PATTERN.match(stripped)
        if m:
            ts = parse_time_to_seconds(m.group(1))
            text = stripped[m.end():].strip()
            parsed.append((ts, text))
        else:
            # Line without timestamp (continuation)
            parsed.append((None, stripped))

    if not chapters:
        return _merge_paragraphs(parsed)

    return _merge_with_chapters(parsed, chapters)


def _merge_paragraphs(parsed_lines):
    """Merge parsed transcript lines into flowing paragraphs."""
    paragraphs = []
    current = []

    for _, text in parsed_lines:
        if not text:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(text)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


def _merge_with_chapters(parsed_lines, chapters):
    """Merge transcript lines with chapter headings inserted."""
    result_parts = []
    current_chapter_idx = 0
    current_paragraph = []

    # Track which chapters we've emitted
    emitted_chapters = set()

    for ts, text in parsed_lines:
        # Check if we've reached a new chapter
        if ts is not None:
            while current_chapter_idx < len(chapters):
                ch_start, ch_title = chapters[current_chapter_idx]
                if ts >= ch_start and current_chapter_idx not in emitted_chapters:
                    # Flush current paragraph before chapter heading
                    if current_paragraph:
                        result_parts.append(" ".join(current_paragraph))
                        current_paragraph = []
                    result_parts.append(f"## {ch_title}")
                    emitted_chapters.add(current_chapter_idx)
                    current_chapter_idx += 1
                else:
                    break

        if not text:
            if current_paragraph:
                result_parts.append(" ".join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(text)

    if current_paragraph:
        result_parts.append(" ".join(current_paragraph))

    return "\n\n".join(result_parts)


def build_output(title, metadata, summary, transcript_text):
    """Build the final markdown output."""
    lines = []

    # Frontmatter — keep only fields from KEEP_FRONTMATTER
    lines.append("---")
    for key in KEEP_FRONTMATTER:
        value = metadata.get(key, "")
        if value:
            if key == "url":
                lines.append(f'{key}: "{value}"')
            else:
                lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")

    # Title heading
    lines.append(f"# {title}")
    lines.append("")

    # Summary (if any)
    if summary:
        lines.append(summary)
        lines.append("")

    # Transcript content
    if transcript_text:
        lines.append(transcript_text)
        lines.append("")

    return "\n".join(lines)


def process_file(filepath):
    """Process a single YouTube transcript markdown file.

    Returns (output_text, stats_dict) or (None, stats_dict) if file can't be processed.
    """
    stats = {"chapters": 0, "has_summary": False}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip newlines for easier processing
    lines = [l.rstrip("\n") for l in lines]

    metadata, body_lines = parse_frontmatter(lines)
    if not metadata.get("title"):
        return None, stats

    description_lines, transcript_lines = extract_sections(body_lines)

    title = metadata["title"]
    summary = extract_description_summary(description_lines)
    chapters = extract_chapters(description_lines)
    transcript_text = process_transcript(transcript_lines, chapters)

    stats["chapters"] = len(chapters)
    stats["has_summary"] = bool(summary)

    output = build_output(title, metadata, summary, transcript_text)
    return output, stats


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess YouTube transcript .md files for indexing"
    )
    ap.add_argument("--inputDir", required=True, help="Directory with raw transcript .md files")
    ap.add_argument("--outputDir", required=True, help="Directory to write processed files")
    ap.add_argument("--dryRun", action="store_true", default=False, help="Preview without writing")
    args = ap.parse_args()

    input_dir = args.inputDir
    output_dir = args.outputDir

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return

    md_files = [f for f in os.listdir(input_dir) if f.endswith(".md")]
    md_files.sort()

    logging.info(f"Found {len(md_files)} .md files in {input_dir}")

    if not args.dryRun:
        os.makedirs(output_dir, exist_ok=True)

    total_processed = 0
    total_with_chapters = 0
    total_with_summary = 0
    total_skipped = 0

    for filename in md_files:
        filepath = os.path.join(input_dir, filename)

        try:
            output, stats = process_file(filepath)
        except Exception as e:
            logging.warning(f"Error processing {filename}: {e}")
            total_skipped += 1
            continue

        if output is None:
            logging.warning(f"Skipped {filename}: no title in frontmatter")
            total_skipped += 1
            continue

        total_processed += 1
        if stats["chapters"] > 0:
            total_with_chapters += 1
        if stats["has_summary"]:
            total_with_summary += 1

        if args.dryRun:
            chapter_info = f"{stats['chapters']} chapters" if stats["chapters"] else "no chapters"
            summary_info = "with summary" if stats["has_summary"] else "no summary"
            logging.info(f"[DRY RUN] {filename}: {chapter_info}, {summary_info}")
        else:
            out_path = os.path.join(output_dir, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output)

    action = "Would process" if args.dryRun else "Processed"
    logging.info(
        f"Done. {action} {total_processed}/{total_processed + total_skipped} files. "
        f"With chapters: {total_with_chapters}, with summary: {total_with_summary}, "
        f"skipped: {total_skipped}"
    )


if __name__ == "__main__":
    main()
