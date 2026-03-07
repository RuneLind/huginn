import re


def sanitize_filename(name):
    """Sanitize a string for use as a filename.

    Replaces filesystem-unsafe characters, collapses whitespace,
    and truncates to 200 characters.
    """
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'[\s_]+', ' ', name).strip()
    if len(name) > 200:
        name = name[:200]
    return name or "Untitled"
