"""Extract a small comparison directly from the pinned external artifact."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "research/snapshots/jevbench/results/v1.2/jevbench-v1.2-results.json"
data = json.loads(source.read_text())
keys = {"jev-1.13.0", "semif-qwen3.5-4b", "djev", "winnow-12b", "system-one-open", "openjev-razorback16"}
fields = ("key", "display", "underlying", "repo", "licence", "jevbench_score", "tiers", "calibration")
result = {"scope": "published_external_results_not_reproduced_by_OpenJeff",
          "source_commit": json.loads((root / "research/snapshots/jevbench/commit.json").read_text())["sha"],
          "benchmark_revision": data["revision"], "artifact_generated_utc": data["generated_utc"],
          "systems": [{k: s[k] for k in fields} for s in data["systems"] if s["key"] in keys]}
(root / "research/leaderboard-extract.json").write_text(json.dumps(result, indent=2) + "\n")
