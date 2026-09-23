"""Dependency-injected orchestration, not a production server or trained gate."""
from dataclasses import dataclass
import math
from typing import Protocol
from .calibration import probabilities, validate_scores
from .contracts import DecisionRequest, validate_ir


class Scorer(Protocol):
    scorer_id: str

    def score(self, request: DecisionRequest, intermediate: dict | None = None) -> list[float]: ...


@dataclass(frozen=True)
class CalibratedScorer:
    scorer: Scorer
    temperature: float
    calibration_id: str
    calibrated_scorer_id: str

    def distribution(self, request, intermediate=None):
        if self.scorer.scorer_id != self.calibrated_scorer_id:
            raise ValueError("scorer/calibrator mismatch")
        scores = self.scorer.score(request, intermediate)
        validate_scores(scores)
        if len(scores) != len(request.candidates):
            raise ValueError("scorer omitted or added candidates")
        return probabilities(scores, self.temperature)


def decide(request, first, *, threshold, refiner=None, intermediate=None):
    """Threshold is caller-supplied and must be validated on independent data.

    A refiner conditioned on IR needs a distinct scorer/calibrator from clean AR.
    Backend exceptions propagate; the API layer must return an explicit error.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    p1 = first.distribution(request)
    p = p1
    path = [first.scorer.scorer_id]
    calibrated = first
    refined = False
    if max(p) < threshold and refiner is not None:
        if intermediate is not None:
            validate_ir(intermediate, request)
        p = refiner.distribution(request, intermediate)
        path.append(refiner.scorer.scorer_id)
        calibrated, refined = refiner, True
    winner = max(range(len(p)), key=p.__getitem__)
    accepted = max(p) >= threshold
    entropy = -sum(v * math.log(v) for v in p if v > 0) / math.log(len(p))
    return {"status": "decided" if accepted else "abstained",
            "decision": request.candidates[winner].id if accepted else None,
            "probabilities": {c.id: v for c, v in zip(request.candidates, p)},
            "confidence": max(p), "normalized_entropy": entropy,
            "refined": refined, "model_path": path,
            "calibration_id": calibrated.calibration_id}

