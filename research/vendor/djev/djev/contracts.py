"""Validated wire types and deterministic distribution conversion.

Confidence is normalized entropy concentration, not empirical calibration.
"""
from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

MAX_QUESTIONS = 32
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_STATE_CHARACTERS = 20_000
MAX_INSTRUCTIONS_CHARACTERS = 2_000
MAX_CRITERION_CHARACTERS = 500
MAX_IMAGE_ATTACHMENTS = 6

Description = str | dict[str, JsonValue] | list[JsonValue] | None
State = str | dict[str, JsonValue] | list[JsonValue]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def finite_json(self):
        # JsonValue permits nested JSON structure; JSON itself forbids NaN/Inf.
        json.dumps(self.model_dump(mode="python"), allow_nan=False)
        return self


class NoulCriteria(StrictModel):
    true: Description = None
    false: Description = None


class NoulQuestion(StrictModel):
    type: Literal["noul"] = "noul"
    instructions: Description = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(StrictModel):
    type: Literal["choice"] = "choice"
    instructions: Description = None
    criteria: dict[str, Description] = Field(min_length=1, max_length=MAX_CHOICE_OPTIONS)


class ScoreQuestion(StrictModel):
    type: Literal["score"] = "score"
    instructions: Description = None
    criteria: list[Description] = Field(min_length=2, max_length=MAX_SCORE_LEVELS)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class InferenceOptions(StrictModel):
    samples: int = Field(default=1, ge=1, le=4)
    steps: Literal[1] = 1
    diagnostics: bool = False
    seed: int | None = Field(default=0, description="Fixed initial diffusion canvas; null requests a random canvas. This is not a bitwise GPU determinism guarantee.")
    isolation: Literal["joint", "independent"] = "joint"
    score_mode: Literal["categorical", "independent_levels"] = "categorical"

    @model_validator(mode="after")
    def isolated_score_requires_independent_questions(self):
        if self.score_mode == "independent_levels" and self.isolation != "independent":
            raise ValueError("independent_levels requires isolation=independent")
        return self

    @field_validator("steps", mode="before")
    @classmethod
    def integer_step(cls, value):
        # Literal[1] alone also accepts True because Python equates True and 1.
        if type(value) is not int:
            raise ValueError("steps must be the integer 1")
        return value


def is_image_description(value: Any) -> bool:
    """Data-URL image objects opt into native attachments; other JSON stays data."""
    return (isinstance(value, dict) and isinstance(value.get("image"), str)
            and value["image"].startswith("data:"))


def description_images(value: Description) -> list[str]:
    """Collect attachment occurrences, including structured description children.

    Validate the opt-in object's shape here; image bytes are checked at the
    public request boundary. Repeated bytes still occupy separate image slots.
    """
    if is_image_description(value):
        if set(value) - {"image", "text"} or ("text" in value and not isinstance(value["text"], str)):
            raise ValueError("image descriptions require exactly image and optional string text")
        return [value["image"]]
    children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return [image for child in children for image in description_images(child)]


def question_descriptions(question: Question) -> list[Description]:
    if isinstance(question, NoulQuestion):
        criteria = [] if question.criteria is None else [question.criteria.false, question.criteria.true]
    elif isinstance(question, ChoiceQuestion):
        criteria = list(question.criteria.values())
    else:
        criteria = question.criteria
    return [question.instructions, *criteria]


def request_image_count(request: DjevRequest) -> int:
    return len(getattr(request, "images", ())) + sum(
        len(description_images(value)) for question in request.questions.values()
        for value in question_descriptions(question)
    )


def request_has_images(request: DjevRequest) -> bool:
    return request_image_count(request) > 0


def _description_text(value: Description) -> Description:
    if is_image_description(value):
        description_images(value)  # Reject malformed opt-in objects before counting.
        return value.get("text", "")
    if isinstance(value, dict):
        return {key: _description_text(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_description_text(child) for child in value]
    return value


def character_count(value: Description, *, image_descriptions: bool = True) -> int:
    """Unicode code points; structured values include compact JSON keys/syntax."""
    if image_descriptions:
        value = _description_text(value)
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False))


class DjevRequest(StrictModel):
    """A complete, bounded decision request."""
    state: State
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)
    model: str = Field(default="djev", min_length=1, max_length=128)
    options: InferenceOptions = Field(default_factory=InferenceOptions)
    images: list[str] = Field(default_factory=list, max_length=1,
        description="At most one base64 JPEG, PNG or WebP data URL; 5 MiB decoded, 2048 pixels per side, one frame.")

    @field_validator("model")
    @classmethod
    def known_model(cls, value):
        if value not in {"djev", "djev-latest", "djev-0.1"}:
            raise ValueError("model must be djev, djev-latest or djev-0.1")
        return value

    @field_validator("state")
    @classmethod
    def bounded_state(cls, value):
        if character_count(value, image_descriptions=False) > MAX_STATE_CHARACTERS:
            raise ValueError(f"state exceeds {MAX_STATE_CHARACTERS} Unicode characters")
        return value

    @field_validator("questions")
    @classmethod
    def bounded_descriptions(cls, value):
        for question in value.values():
            if character_count(question.instructions) > MAX_INSTRUCTIONS_CHARACTERS:
                raise ValueError(f"instructions exceed {MAX_INSTRUCTIONS_CHARACTERS} Unicode characters")
            if isinstance(question, NoulQuestion):
                criteria = [] if question.criteria is None else [question.criteria.true, question.criteria.false]
            elif isinstance(question, ChoiceQuestion):
                criteria = [*question.criteria.keys(), *question.criteria.values()]
            else:
                criteria = question.criteria
            if any(character_count(item) > MAX_CRITERION_CHARACTERS for item in criteria):
                raise ValueError(f"criterion descriptions and choice names cannot exceed {MAX_CRITERION_CHARACTERS} Unicode characters")
        return value

    @field_validator("images")
    @classmethod
    def validated_images(cls, value):
        from .images import validate_image_data_url
        for item in value:
            validate_image_data_url(item)
        return value

    @model_validator(mode="after")
    def validated_question_images(self):
        from .images import validate_image_data_url
        if request_image_count(self) > MAX_IMAGE_ATTACHMENTS:
            raise ValueError(f"at most {MAX_IMAGE_ATTACHMENTS} image attachments are supported, including state images and repeated attachments")
        for question in self.questions.values():
            for value in question_descriptions(question):
                for image in description_images(value):
                    validate_image_data_url(image)
        return self


def normalize_logprobs(logprobs: list[float]) -> list[float]:
    """Stable softmax over a closed set, preserving impossible (-Inf) labels.

    Missing or non-finite evidence must fail rather than fabricate a uniform
    distribution. Positive infinity and NaN are invalid backend outputs.
    """
    if not logprobs:
        raise ValueError("at least one log probability is required")
    values = [float(value) for value in logprobs]
    if any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError("log probabilities cannot contain NaN or positive infinity")
    peak = max(values)
    if peak == -math.inf:
        raise ValueError("at least one log probability must be finite")
    weights = [math.exp(value - peak) for value in values]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _validated_probabilities(probabilities: list[float], count: int) -> list[float]:
    if len(probabilities) != count:
        raise ValueError(f"expected {count} probabilities, received {len(probabilities)}")
    values = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("probabilities must be finite numbers between 0 and 1")
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("probabilities must sum to 1")
    return [value / total for value in values]


def _entropy_concentration(probabilities: list[float]) -> float:
    if len(probabilities) == 1:
        return 1.0
    entropy = -math.fsum(value * math.log(value) for value in probabilities if value > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(probabilities))))


def answer_from_probabilities(question: Question, probabilities: list[float]) -> dict[str, Any]:
    """Build a typed answer from a complete, normalized model distribution.

    Noul uses [false, true]. Choice follows criteria insertion order. Score
    follows array order and returns the expected zero-based level index.
    """
    if isinstance(question, NoulQuestion):
        values = _validated_probabilities(probabilities, 2)
        return {"type": "noul", "noul": values[1]}
    if isinstance(question, ChoiceQuestion):
        labels = list(question.criteria)
        values = _validated_probabilities(probabilities, len(labels))
        winner = max(range(len(values)), key=values.__getitem__)
        return {
            "type": "choice",
            "choice": labels[winner],
            "probabilities": dict(zip(labels, values, strict=True)),
            "confidence": _entropy_concentration(values),
        }
    if isinstance(question, ScoreQuestion):
        values = _validated_probabilities(probabilities, len(question.criteria))
        return {
            "type": "score",
            "score": math.fsum(index * value for index, value in enumerate(values)),
            "legend": {str(index): description for index, description in enumerate(question.criteria)},
            "probabilities": {str(index): value for index, value in enumerate(values)},
            "confidence": _entropy_concentration(values),
        }
    raise TypeError("unsupported question type")
