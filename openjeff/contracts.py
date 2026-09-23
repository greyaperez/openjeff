"""Small typed research contract; all state is data, never tool instructions."""
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class Candidate:
    id: str
    description: str


@dataclass(frozen=True)
class DecisionRequest:
    state: object
    question: str
    candidates: tuple[Candidate, ...]
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("nonempty question required")
        if not 2 <= len(self.candidates) <= 26:
            raise ValueError("research code alphabet supports 2–26 candidates")
        ids = [c.id for c in self.candidates]
        if len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("candidate IDs must be nonempty and unique")
        if any(not isinstance(c.description, str) or not c.description for c in self.candidates):
            raise ValueError("candidate descriptions required")
        if len(set(self.evidence_ids)) != len(self.evidence_ids) or any(not isinstance(x, str) or not x for x in self.evidence_ids):
            raise ValueError("evidence IDs must be nonempty and unique")
        encoded = json.dumps(self.state, allow_nan=False)
        if len(encoded.encode()) > 128_000:
            raise ValueError("state exceeds research byte limit")
        if len(self.question) + sum(len(c.description) for c in self.candidates) > 32_000:
            raise ValueError("rubric exceeds research character limit")


def validate_ir(ir, request):
    allowed = {"schema_version", "candidate_support", "uncertainty_flags", "ambiguous_dimensions", "refinement_targets"}
    if not isinstance(ir, dict) or set(ir) != allowed or ir["schema_version"] != "openjeff.ir.v1":
        raise ValueError("invalid IR schema")
    if len(json.dumps(ir, allow_nan=False).encode()) > 4096:
        raise ValueError("IR byte limit exceeded; backend must also enforce its token limit")
    evidence = set(request.evidence_ids)
    candidates = {c.id for c in request.candidates}
    support = ir["candidate_support"]
    if not isinstance(support, list) or len(support) > len(candidates):
        raise ValueError("invalid candidate support list")
    seen = set()
    for record in support:
        if not isinstance(record, dict) or set(record) != {"candidate_id", "support_ids", "conflict_ids"}:
            raise ValueError("invalid candidate evidence record")
        cid = record["candidate_id"]
        if not isinstance(cid, str) or cid not in candidates or cid in seen:
            raise ValueError("unknown or repeated candidate")
        seen.add(cid)
        for field in ("support_ids", "conflict_ids"):
            refs = record[field]
            if not isinstance(refs, list) or len(refs) > 16 or any(not isinstance(r, str) or r not in evidence for r in refs):
                raise ValueError("unknown or invalid evidence reference")
    targets = ir["refinement_targets"]
    if not isinstance(targets, list) or len(targets) > 16 or any(not isinstance(r, str) or r not in evidence for r in targets):
        raise ValueError("invalid refinement targets")
    flags = ir["uncertainty_flags"]
    flag_set = {"conflicting_evidence", "missing_evidence", "ambiguous_rubric", "out_of_distribution"}
    if not isinstance(flags, list) or len(flags) > 4 or any(not isinstance(f, str) or f not in flag_set for f in flags):
        raise ValueError("invalid uncertainty flags")
    dims = ir["ambiguous_dimensions"]
    if not isinstance(dims, list) or len(dims) > 8 or any(not isinstance(d, str) or len(d) > 64 for d in dims):
        raise ValueError("invalid ambiguous dimensions")
    return ir

