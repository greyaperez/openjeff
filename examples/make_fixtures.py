"""Deliberately artificial scores for numerical smoke tests, NOT model outputs."""
import json
from pathlib import Path
from openjeff.calibration import digest

root = Path(__file__).parent
for name, split in [("calibration", "calibration_fit"), ("evaluation", "calibration_check")]:
    rows = [{"id": f"{name}-{i}", "group_id": f"{name}-group-{i}",
             "input_sha256": digest(f"synthetic fixture {name} input {i}"),
             "split": split, "scorer_id": "synthetic-fixture-not-a-model",
             "scores": [10.0, 0.0], "target": 0 if i % 4 else 1,
             "source": "hand_constructed_numerical_fixture"} for i in range(12)]
    (root / f"{name}_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

