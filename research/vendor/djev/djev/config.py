"""Small, explicit settings shared by the API and pinned GPU runtime."""
import os

MODEL = "google/diffusiongemma-26B-A4B-it"
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
MAX_BODY_BYTES = 8 * 1024 * 1024


def bounded_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def get_max_model_len() -> int:
    return bounded_integer("DJEV_MAX_MODEL_LEN", 32768, 1024, 32768)


def get_max_request_reads() -> int:
    return bounded_integer("DJEV_MAX_REQUEST_READS", 4, 1, 8)
