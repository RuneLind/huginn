"""Environment-variable helpers."""

import os


_TRUTHY = ("1", "true", "yes")


def env_bool(name, default=False):
    """Read a boolean-ish env var. Truthy values: "1", "true", "yes" (case-insensitive).

    Returns `default` when the var is unset; any other value is treated as False.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in _TRUTHY
