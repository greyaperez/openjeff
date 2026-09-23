"""Public boundary tests: type guarantees, probability math, and bounded work."""

import math

import pytest
from pydantic import ValidationError

from djev.contracts import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    DjevRequest,
    answer_from_probabilities,
    normalize_logprobs,
)


def request(**changes):
    body = {"state": "Refund the duplicate charge.", "questions": {"q": {"type": "noul"}}}
    body.update(changes)
    return DjevRequest.model_validate(body)


def test_accepts_structured_questions_without_rewriting_external_ids():
    parsed = request(
        state={"ticket": ["Refund please", {"amount": 42, "paid": True}]},
        questions={
            "identificador-ñ": {
                "type": "choice",
                "instructions": {"question": "Which team?", "exceptions": ["Do not guess"]},
                "criteria": {"facturación": {"examples": ["Refunds"]}, "other": None},
            },
            "severity": {"type": "score", "criteria": [{"meaning": "Low"}, ["High"]]},
            "refund": {"type": "noul", "criteria": {"true": ["Explicit request"], "false": None}},
        },
    )
    assert list(parsed.questions) == ["identificador-ñ", "severity", "refund"]
    assert parsed.questions["identificador-ñ"].criteria["facturación"] == {"examples": ["Refunds"]}
    assert parsed.questions["severity"].criteria == [{"meaning": "Low"}, ["High"]]
    assert parsed.questions["refund"].criteria.true == ["Explicit request"]
    assert parsed.model == "djev"
    assert parsed.options.model_dump() == {"samples": 1, "steps": 1, "diagnostics": False, "seed": 0, "isolation": "joint", "score_mode": "categorical"}


def test_question_isolation_is_explicit_and_joint_remains_default():
    assert request().options.isolation == "joint"
    assert request(options={"isolation": "independent"}).options.isolation == "independent"
    for invalid in (True, None, "question", "automatic"):
        with pytest.raises(ValidationError):
            request(options={"isolation": invalid})


@pytest.mark.parametrize("state", [None, True, 42, 0.2])
def test_rejects_non_content_root_state(state):
    with pytest.raises(ValidationError):
        request(state=state)


@pytest.mark.parametrize("body", [
    {"questions": {}},
    {"questions": {"q": {"type": "text", "instructions": "Respond"}}},
    {"questions": {"q": {"type": "choice"}}},
    {"questions": {"q": {"type": "choice", "criteria": {}}}},
    {"questions": {"q": {"type": "score", "criteria": ["Only one"]}}},
    {"questions": {"q": {"type": "noul", "criteria": {"maybe": "Uncertain"}}}},
    {"questions": {"q": {"type": "noul", "unexpected": True}}},
    {"unexpected": "ignored?"},
    {"options": {"samples": 0}},
    {"options": {"samples": 5}},
    {"options": {"steps": 2}},
    {"options": {"steps": True}},
    {"options": {"seed": True}},
    {"options": {"samples": "2"}},
    {"options": {"diagnostics": "true"}},
])
def test_rejects_invalid_or_unsupported_work(body):
    with pytest.raises(ValidationError):
        request(**body)


def test_work_limits_accept_boundary_and_reject_oversize():
    request(questions={str(i): {"type": "noul"} for i in range(32)})
    with pytest.raises(ValidationError):
        request(questions={str(i): {"type": "noul"} for i in range(33)})
    ChoiceQuestion(criteria={str(i): None for i in range(255)})
    with pytest.raises(ValidationError):
        ChoiceQuestion(criteria={str(i): None for i in range(256)})
    ScoreQuestion(criteria=[str(i) for i in range(10)])
    with pytest.raises(ValidationError):
        ScoreQuestion(criteria=[str(i) for i in range(11)])


@pytest.mark.parametrize("body", [
    {"state": {"nested": [float("nan")]}},
    {"questions": {"q": {"type": "noul", "instructions": [float("inf")]}}},
    {"questions": {"q": {"type": "choice", "criteria": {"a": {"v": float("-inf")}}}}},
])
def test_rejects_nonfinite_nested_json(body):
    with pytest.raises(ValidationError):
        request(**body)


def test_softmax_is_stable_at_extreme_magnitudes():
    assert normalize_logprobs([10000.0, 10000.0, -10000.0]) == pytest.approx([0.5, 0.5, 0.0])
    assert normalize_logprobs([math.log(2), 0.0, -math.inf]) == pytest.approx([2 / 3, 1 / 3, 0.0])


@pytest.mark.parametrize("values", [[], [math.nan], [math.inf, 0.0], [-math.inf, -math.inf]])
def test_softmax_rejects_missing_or_unusable_evidence(values):
    with pytest.raises(ValueError):
        normalize_logprobs(values)


def test_noul_uses_probability_of_true_without_extra_confidence():
    assert answer_from_probabilities(NoulQuestion(), [0.2, 0.8]) == {"type": "noul", "noul": 0.8}


def test_choice_preserves_labels_and_resolves_tie_in_input_order():
    question = ChoiceQuestion(criteria={"primero": None, "segundo": {"description": "Second"}})
    assert answer_from_probabilities(question, [0.5, 0.5]) == {
        "type": "choice", "choice": "primero",
        "probabilities": {"primero": 0.5, "segundo": 0.5}, "confidence": 0.0,
    }


def test_score_is_expectation_and_echoes_structured_legend():
    question = ScoreQuestion(criteria=[{"meaning": "Low"}, "Medium", ["High"]])
    result = answer_from_probabilities(question, [0.25, 0.5, 0.25])
    assert result["score"] == 1.0
    assert result["legend"] == {"0": {"meaning": "Low"}, "1": "Medium", "2": ["High"]}
    assert result["probabilities"] == {"0": 0.25, "1": 0.5, "2": 0.25}
    assert result["confidence"] == pytest.approx(0.053605369642814)


def test_one_hot_and_singleton_have_maximum_concentration():
    assert answer_from_probabilities(ScoreQuestion(criteria=["Low", "High"]), [0.0, 1.0])["confidence"] == 1.0
    assert answer_from_probabilities(ChoiceQuestion(criteria={"only": None}), [1.0])["confidence"] == 1.0


@pytest.mark.parametrize("probabilities", [[], [1.0], [0.2, 0.3, 0.5], [-0.1, 1.1], [0.0, 0.0], [1.0, 1.0], [math.nan, 1.0], [math.inf, 0.0]])
def test_answer_conversion_rejects_invalid_distributions(probabilities):
    with pytest.raises(ValueError):
        answer_from_probabilities(NoulQuestion(), probabilities)
