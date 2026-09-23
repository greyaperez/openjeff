"""Dependency-free score calibration and leakage checks for offline experiments."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_scores(scores):
    if not isinstance(scores, (list, tuple)) or len(scores) < 2:
        raise ValueError("at least two candidate scores are required")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or
           not math.isfinite(x) or abs(x) > 1e6 for x in scores):
        raise ValueError("scores must be finite numeric values with magnitude <= 1e6")


def log_probabilities(scores, temperature=1.0):
    validate_scores(scores)
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    m = max(scores)
    z = [(s - m) / temperature for s in scores]
    norm = math.log(sum(math.exp(v) for v in z))
    return [v - norm for v in z]


def probabilities(scores, temperature=1.0):
    return [math.exp(v) for v in log_probabilities(scores, temperature)]


def validate_rows(rows):
    if not rows:
        raise ValueError("empty score dataset")
    seen = set()
    for row in rows:
        validate_scores(row["scores"])
        for field in ("id", "group_id", "split", "scorer_id", "input_sha256"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"missing nonempty {field}")
        if len(row["input_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["input_sha256"]):
            raise ValueError("input_sha256 must be a lowercase SHA-256")
        if row["id"] in seen:
            raise ValueError("duplicate row id")
        seen.add(row["id"])
        target = row.get("target")
        if type(target) is not int or not 0 <= target < len(row["scores"]):
            raise ValueError("target must be an in-range integer candidate index")
    if len({r["scorer_id"] for r in rows}) != 1:
        raise ValueError("one frozen scorer configuration per dataset")


def mean_nll(rows, temperature):
    return sum(-log_probabilities(r["scores"], temperature)[r["target"]]
               for r in rows) / len(rows)


@dataclass(frozen=True)
class Calibration:
    temperature: float
    scorer_id: str
    fit_data_sha256: str
    fit_group_hashes: tuple
    fit_input_hashes: tuple
    fit_row_hashes: tuple
    n_fit: int
    method: str = "bounded_log_temperature_search_v1"

    def to_dict(self):
        return asdict(self)

    @property
    def artifact_id(self):
        return digest(self.to_dict())

    def validate_evaluation(self, rows):
        validate_rows(rows)
        if rows[0]["scorer_id"] != self.scorer_id:
            raise ValueError("calibrator and scorer configuration mismatch")
        if any(r["split"] not in {"calibration_check", "adversarial", "production_sim", "final", "test"}
               for r in rows):
            raise ValueError("evaluate only an explicitly held-out split")
        if set(self.fit_group_hashes) & {digest(r["group_id"]) for r in rows}:
            raise ValueError("calibration/evaluation scenario groups overlap")
        if set(self.fit_input_hashes) & {r["input_sha256"] for r in rows}:
            raise ValueError("calibration/evaluation inputs overlap")
        if set(self.fit_row_hashes) & {digest(r["id"]) for r in rows}:
            raise ValueError("calibration/evaluation row IDs overlap")


def fit_temperature(rows):
    validate_rows(rows)
    if any(r["split"] != "calibration_fit" for r in rows):
        raise ValueError("temperature can only be fitted on calibration_fit")
    # Bounded grid followed by local refinement. Include T=1 as an exact control.
    lo, hi = math.log(0.05), math.log(20.0)
    grid = [lo + (hi - lo) * i / 120 for i in range(121)] + [0.0]
    best = min(grid, key=lambda x: mean_nll(rows, math.exp(x)))
    width = (hi - lo) / 120
    for _ in range(8):
        left, right = max(lo, best - width), min(hi, best + width)
        values = [left + (right - left) * i / 10 for i in range(11)] + [best, 0.0]
        best = min(values, key=lambda x: mean_nll(rows, math.exp(x)))
        width /= 4
    return Calibration(math.exp(best), rows[0]["scorer_id"], digest(rows),
                       tuple(sorted({digest(r["group_id"]) for r in rows})),
                       tuple(sorted({r["input_sha256"] for r in rows})),
                       tuple(sorted({digest(r["id"]) for r in rows})), len(rows))

