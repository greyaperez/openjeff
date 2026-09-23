"""Offline metrics. These are internal metrics, not the JevBench composite."""
import math
import random
from .calibration import log_probabilities, probabilities, validate_rows


def evaluate(rows, temperature=1.0, bins=15):
    validate_rows(rows)
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    records = []
    nll = brier = 0.0
    for r in rows:
        p = probabilities(r["scores"], temperature)
        pred = max(range(len(p)), key=p.__getitem__)
        correct = int(pred == r["target"])
        confidence = p[pred]
        nll -= log_probabilities(r["scores"], temperature)[r["target"]]
        brier += sum((v - int(i == r["target"])) ** 2 for i, v in enumerate(p))
        records.append((confidence, correct))
    n = len(rows)
    reliability = []
    ece = 0.0
    for b in range(bins):
        bucket = [(c, ok) for c, ok in records if min(bins - 1, int(c * bins)) == b]
        if bucket:
            confidence = sum(c for c, _ in bucket) / len(bucket)
            accuracy = sum(ok for _, ok in bucket) / len(bucket)
            ece += len(bucket) / n * abs(confidence - accuracy)
        else:
            confidence = accuracy = None
        reliability.append({"lower": b / bins, "upper": (b + 1) / bins,
                            "count": len(bucket), "confidence": confidence, "accuracy": accuracy})
    # Accept entire confidence ties together. Arbitrary tie-breaking inflates coverage claims.
    curve = []
    ordered = sorted(records, reverse=True)
    accepted = correct_count = 0
    index = 0
    while index < n:
        threshold = ordered[index][0]
        while index < n and ordered[index][0] == threshold:
            accepted += 1
            correct_count += ordered[index][1]
            index += 1
        curve.append({"threshold": threshold, "coverage": accepted / n,
                      "risk": 1 - correct_count / accepted, "n_accepted": accepted})
    high = [(c, ok) for c, ok in records if c >= 0.9]
    return {"n": n, "accuracy": sum(ok for _, ok in records) / n,
            "nll": nll / n, "brier_sum_over_classes": brier / n, "ece": ece,
            "wrong_high_confidence_fraction": sum(c >= 0.9 and not ok for c, ok in records) / n,
            "error_given_high_confidence": (sum(not ok for _, ok in high) / len(high)) if high else None,
            "correct_low_confidence_fraction": sum(c < 0.6 and ok for c, ok in records) / n,
            "reliability": reliability, "risk_coverage": curve}


def refinement_metrics(first_correct, final_correct):
    a, b = list(first_correct), list(final_correct)
    if not a or len(a) != len(b) or any(type(v) is not bool for v in a + b):
        raise ValueError("equal nonempty Boolean correctness arrays required")
    n = len(a)
    gain = sum(not x and y for x, y in zip(a, b))
    damage = sum(x and not y for x, y in zip(a, b))
    wrong = n - sum(a)
    right = sum(a)
    e1 = [int(not x) for x in a]
    e2 = [int(not x) for x in b]
    m1, m2 = sum(e1) / n, sum(e2) / n
    denom = math.sqrt(sum((v - m1) ** 2 for v in e1) * sum((v - m2) ** 2 for v in e2))
    phi = sum((x - m1) * (y - m2) for x, y in zip(e1, e2)) / denom if denom else None
    return {"n": n, "gain": gain / n, "damage": damage / n,
            "net_accuracy_change": (gain - damage) / n,
            "repair_given_wrong": gain / wrong if wrong else None,
            "break_given_right": damage / right if right else None,
            "error_phi": phi, "oracle_union_accuracy": sum(x or y for x, y in zip(a, b)) / n}


def paired_accuracy_interval(first_correct, final_correct, resamples=2000, seed=23):
    """Row bootstrap ONLY for independent examples. Cluster related scenarios externally."""
    refinement_metrics(first_correct, final_correct)
    if type(resamples) is not int or resamples < 100:
        raise ValueError("use at least 100 resamples")
    differences = [int(b) - int(a) for a, b in zip(first_correct, final_correct)]
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(sum(rng.choice(differences) for _ in range(n)) / n for _ in range(resamples))
    return {"lower": samples[int(0.025 * resamples)],
            "upper": samples[min(resamples - 1, math.ceil(0.975 * resamples) - 1)],
            "method": "paired_row_bootstrap_independent_items_only", "seed": seed}

