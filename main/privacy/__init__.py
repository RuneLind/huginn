"""Build-time people aliasing; see alias_registry.py."""
from pathlib import Path

# Where every privacy output that carries REAL TEXT defaults to: the scan's
# candidate pairs and the sensitivity sweep's report. `data/` is gitignored
# wholesale, so the default cannot land a list of real names in a tracked file.
# It lives here rather than in either writer because both need it and neither
# owns it — and because a second copy of this path is a second chance to spell
# it as somewhere git tracks.
DEFAULT_PRIVACY_DIR = Path(__file__).resolve().parents[2] / "data" / "privacy"
