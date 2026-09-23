"""Recompute diagnostic results and an explicitly post-hoc order-averaging check."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from openjeff.calibration import probabilities
from openjeff.evaluation import evaluate
from openjeff.probe_cases import generate_cases

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/gpu-pilot-2026-09-23"


def equivalent(left, right):
    if isinstance(left, float) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(v, right[k]) for k, v in left.items())
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    return left == right


def summarize():
    cases = {case["id"]: case for case in generate_cases()}
    results = []
    for path in sorted((RUN / "reports").glob("gemma-*/summary.json")):
        summary = json.loads(path.read_text())
        score_file = path.parent / "scores.jsonl"
        assert hashlib.sha256(score_file.read_bytes()).hexdigest() == summary["scores_sha256"]
        rows = [json.loads(line) for line in score_file.read_text().splitlines()]
        assert len(rows) == len(cases) and {r["id"] for r in rows} == set(cases)
        family_rows, order_rows, group_rows = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in rows:
            case = cases[row["id"]]
            assert row["target"] == case["target"] and row["input_sha256"] == case["input_sha256"]
            assert row["family"] == case["family"] and row["group_id"] == case["group_id"]
            family_rows[row["family"]].append(row)
            order_rows["reversed" if row["id"].endswith("reversed") else "original"].append(row)
            group_rows[row["group_id"]].append(row)
        metrics = evaluate(rows)
        # Python 3.12's improved float summation differs by a few ULPs from 3.10.
        assert equivalent(metrics, summary["metrics"])
        compact = {key: value for key, value in metrics.items() if key not in {"reliability", "risk_coverage"}}
        pooled_correct = 0
        for pair in group_rows.values():
            assert len(pair) == 2
            aligned = defaultdict(float)
            for row in pair:
                candidates = cases[row["id"]]["request"]["candidates"]
                for candidate, probability in zip(candidates, probabilities(row["scores"])):
                    aligned[candidate["id"]] += probability / 2
            winner = max(sorted(aligned), key=aligned.__getitem__)
            pooled_correct += winner == cases[pair[0]["id"]]["answer_id"]
        results.append({"model": summary["model"], "metrics": compact,
                        "serial_latency_s": summary["serial_latency_s"],
                        "peak_allocated_gib": summary["device_peak_allocated_bytes"] / 2**30,
                        "option_order_disagreement": summary["option_order_disagreement"],
                        "family_accuracy": {k: evaluate(v)["accuracy"] for k, v in family_rows.items()},
                        "order_accuracy": {k: evaluate(v)["accuracy"] for k, v in order_rows.items()},
                        "post_hoc_two_order_mean_probability": {
                            "accuracy": pooled_correct / len(group_rows), "groups": len(group_rows),
                            "forward_passes_per_scenario": 2, "new_gpu_inference": False,
                            "limitation": "Exploratory reuse of development predictions; not independent validation or calibration."}})
    output = {"scope": "compatibility_diagnostics_only", "scenario_groups": 100,
              "rows_per_model": 200, "models": results}
    (RUN / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    summarize()
