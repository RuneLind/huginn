"""Benchmark result recording, persistence, and comparison."""

import json
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    category: str
    metrics: dict[str, float]
    duration_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    """Collection of benchmark results from a single run."""
    timestamp: str
    git_commit: str
    system_info: dict
    results: list[BenchmarkResult]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "system_info": self.system_info,
            "results": [asdict(r) for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "BenchmarkRun":
        data = json.loads(json_str)
        return cls(
            timestamp=data["timestamp"],
            git_commit=data["git_commit"],
            system_info=data["system_info"],
            results=[BenchmarkResult(**r) for r in data["results"]],
        )

    def save(self, results_dir: Path) -> Path:
        """Save run to timestamped JSON file and update latest symlink."""
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = self.timestamp.replace(":", "-").replace("T", "_") + ".json"
        filepath = results_dir / filename
        filepath.write_text(self.to_json())

        latest = results_dir / "latest.json"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(filename)

        return filepath


def get_system_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def create_run(results: list[BenchmarkResult]) -> BenchmarkRun:
    return BenchmarkRun(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=get_git_commit(),
        system_info=get_system_info(),
        results=results,
    )


def load_latest(results_dir: Path) -> BenchmarkRun | None:
    latest = results_dir / "latest.json"
    if not latest.exists():
        return None
    return BenchmarkRun.from_json(latest.read_text())


def compare_runs(current: BenchmarkRun, baseline: BenchmarkRun, tolerance: float = 0.10) -> dict:
    """Compare two runs and categorize changes.

    Returns dict with 'regressions', 'improvements', and 'stable' lists.
    Each entry: {"benchmark": name, "metric": key, "current": val, "baseline": val, "delta_pct": pct}.
    """
    baseline_lookup = {}
    for r in baseline.results:
        for k, v in r.metrics.items():
            baseline_lookup[(r.name, k)] = v

    regressions = []
    improvements = []
    stable = []

    for r in current.results:
        for k, v in r.metrics.items():
            bv = baseline_lookup.get((r.name, k))
            if bv is None:
                continue

            entry = {
                "benchmark": r.name,
                "metric": k,
                "current": v,
                "baseline": bv,
            }

            if abs(bv) < 1e-9:
                entry["delta_pct"] = 0.0
                stable.append(entry)
                continue

            # For metrics where higher is better (recall, mrr, hit_rate, f1, etc.)
            # regression = current < baseline
            # For latency metrics (contains _ms, _us, _seconds), lower is better
            # regression = current > baseline
            is_latency = any(s in k for s in ("_ms", "_us", "_seconds", "latency", "load_"))
            delta_pct = (v - bv) / abs(bv)
            entry["delta_pct"] = delta_pct

            if is_latency:
                if delta_pct > tolerance:
                    regressions.append(entry)
                elif delta_pct < -tolerance:
                    improvements.append(entry)
                else:
                    stable.append(entry)
            else:
                if delta_pct < -tolerance:
                    regressions.append(entry)
                elif delta_pct > tolerance:
                    improvements.append(entry)
                else:
                    stable.append(entry)

    return {"regressions": regressions, "improvements": improvements, "stable": stable}


def format_summary(run: BenchmarkRun, comparison: dict | None = None) -> str:
    """Format a human-readable summary table."""
    lines = []
    lines.append(f"Benchmark Run: {run.timestamp}  (git: {run.git_commit})")
    lines.append(f"Platform: {run.system_info.get('platform', 'unknown')}")
    lines.append("")

    # Build delta lookup
    deltas = {}
    if comparison:
        for cat in ("regressions", "improvements", "stable"):
            for entry in comparison[cat]:
                key = (entry["benchmark"], entry["metric"])
                deltas[key] = (entry["delta_pct"], cat)

    # Group results by category
    by_category: dict[str, list[BenchmarkResult]] = {}
    for r in run.results:
        by_category.setdefault(r.category, []).append(r)

    for category, results in sorted(by_category.items()):
        lines.append(f"== {category.upper()} ==")
        for r in results:
            lines.append(f"  {r.name}  ({r.duration_ms:.0f}ms)")
            for k, v in sorted(r.metrics.items()):
                delta_str = ""
                delta_info = deltas.get((r.name, k))
                if delta_info:
                    pct, cat = delta_info
                    if cat == "regressions":
                        delta_str = f"  \u25bc {pct:+.1%}"
                    elif cat == "improvements":
                        delta_str = f"  \u25b2 {pct:+.1%}"
                    else:
                        delta_str = f"  = {pct:+.1%}"

                if isinstance(v, float):
                    if v > 100:
                        lines.append(f"    {k}: {v:.0f}{delta_str}")
                    else:
                        lines.append(f"    {k}: {v:.4f}{delta_str}")
                else:
                    lines.append(f"    {k}: {v}{delta_str}")
        lines.append("")

    if comparison:
        n_reg = len(comparison["regressions"])
        n_imp = len(comparison["improvements"])
        n_stab = len(comparison["stable"])
        lines.append(f"Summary: {n_imp} improvements, {n_reg} regressions, {n_stab} stable")

    return "\n".join(lines)
