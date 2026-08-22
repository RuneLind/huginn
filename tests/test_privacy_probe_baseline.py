"""The probe set is the campaign's before-picture, so it must actually contain
the two shapes a naive substituter misses: a `require_full_name` entry and one
attested in the deactivated-account `Name [X]` form. Invented names only."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "audit"))

import privacy_probe_baseline  # noqa: E402


def _entry(alias, name, role="dev", require_full_name=False):
    return {"alias": alias, "name": name, "role": role, "require_full_name": require_full_name}


# One role, so the role spread cannot pull the two special entries in, and both
# of them are attested least often — they are only reachable via the top-ups,
# which is exactly where the truncation dropped the first of the two.
MAP = {"entries": [
    _entry("dev-01", "Ada Alpha"),
    _entry("dev-02", "Ada Beta"),
    _entry("dev-03", "Ada Gamma"),
    _entry("fag-01", "Bo Tester", require_full_name=True),
    _entry("arch-01", "Zylphia Quorndal"),
]}

CORPUS = ("Ada Alpha " * 30 + "Ada Beta " * 20 + "Ada Gamma " * 10
          + "Bo Tester " + "Zylphia Quorndal [X] ")


def test_both_required_shapes_survive_the_truncation():
    probes = privacy_probe_baseline.pick_probes(MAP, CORPUS, size=3)
    assert len(probes) == 3
    assert len({p["alias"] for p in probes}) == 3            # no duplicates
    assert any(p["requireFullName"] for p in probes)
    assert any(p["bracketHits"] > 0 for p in probes)


def test_unattested_entries_are_never_probed():
    probes = privacy_probe_baseline.pick_probes(
        {"entries": [_entry("dev-01", "Ada Alpha"), _entry("dev-99", "Never Mentioned")]},
        CORPUS, size=5)
    assert [p["alias"] for p in probes] == ["dev-01"]
