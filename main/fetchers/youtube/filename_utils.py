"""
Filename utilities for YouTube fetcher.

This module provides utilities for creating safe, consistent filenames
for YouTube videos, following the pattern: {video_id}_{safe_title}.{ext}
"""

import re
import unicodedata


def create_safe_filename(video_id: str, title: str, extension: str) -> str:
    """
    Create a safe filename from video ID and title.

    Format: {video_id}_{safe_title}.{extension}

    Args:
        video_id: YouTube video ID (will be kept as-is)
        title: Video title (will be sanitized)
        extension: File extension without dot (e.g., "md", "json")

    Returns:
        Safe filename string

    Examples:
        >>> create_safe_filename("abc123", "How to Build RAG Systems!", "md")
        'abc123_how_to_build_rag_systems.md'

        >>> create_safe_filename("xyz789", "Testing & Validation", "json")
        'xyz789_testing_validation.json'
    """
    safe_title = sanitize_title(title, max_length=100)

    # Ensure extension doesn't have leading dot
    extension = extension.lstrip(".")

    return f"{video_id}_{safe_title}.{extension}"


def sanitize_title(title: str, max_length: int = 100) -> str:
    """
    Sanitize a title to make it safe for use in filenames.

    Rules:
    - Convert to lowercase
    - Replace spaces with underscores
    - Remove special characters (keep only alphanumeric and underscores)
    - Normalize unicode characters to ASCII equivalents
    - Limit length
    - Remove leading/trailing underscores

    Args:
        title: Original title
        max_length: Maximum length of sanitized title (default: 100)

    Returns:
        Sanitized title safe for filenames

    Examples:
        >>> sanitize_title("How to Build RAG Systems!")
        'how_to_build_rag_systems'

        >>> sanitize_title("Testing & Validation")
        'testing_validation'

        >>> sanitize_title("Café — Special Édition")
        'cafe_special_edition'
    """
    # Normalize unicode to ASCII (e.g., é -> e, å -> a)
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii")

    # Convert to lowercase
    title = title.lower()

    # Replace spaces and common separators with underscores
    title = re.sub(r"[\s\-—–]+", "_", title)

    # Remove special characters, keep only alphanumeric and underscores
    title = re.sub(r"[^a-z0-9_]", "", title)

    # Replace multiple consecutive underscores with single underscore
    title = re.sub(r"_+", "_", title)

    # Limit length
    title = title[:max_length]

    # Remove leading/trailing underscores
    title = title.strip("_")

    # Ensure we have at least something
    if not title:
        title = "untitled"

    return title


def validate_filename(filename: str) -> bool:
    """
    Validate that a filename is safe for cross-platform use.

    Args:
        filename: Filename to validate

    Returns:
        True if filename is safe, False otherwise
    """
    # Check for invalid characters (varies by OS, using most restrictive set)
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    if re.search(invalid_chars, filename):
        return False

    # Check for reserved names (Windows)
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
    name_without_ext = filename.rsplit(".", 1)[0].upper()
    if name_without_ext in reserved_names:
        return False

    # Check length (most file systems have 255 char limit)
    if len(filename) > 255:
        return False

    return True


if __name__ == "__main__":
    # Run doctests
    import doctest

    doctest.testmod()
