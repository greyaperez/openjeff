"""Validate user-supplied image bytes locally; no network or URL resolution."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from io import BytesIO
import warnings

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048
IMAGE_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


@dataclass(frozen=True)
class ValidatedImage:
    mime_type: str
    width: int
    height: int
    data: bytes = field(repr=False)


def validate_image_data_url(value: str) -> ValidatedImage:
    if not isinstance(value, str):
        raise ValueError("image must be a base64 JPEG, PNG or WebP data URL")
    prefix, separator, encoded = value.partition(",")
    accepted = {f"data:{mime};base64": mime for mime in IMAGE_FORMATS}
    if not separator or prefix not in accepted:
        raise ValueError("image must be a base64 JPEG, PNG or WebP data URL; remote URLs are not accepted")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} decoded bytes")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("image data must contain valid base64") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image must contain between 1 and {MAX_IMAGE_BYTES} decoded bytes")
    from PIL import Image, UnidentifiedImageError
    mime = accepted[prefix]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as picture:
                if picture.format != IMAGE_FORMATS[mime]:
                    raise ValueError("image MIME type does not match its actual format")
                width, height = picture.size
                if not (1 <= width <= MAX_IMAGE_DIMENSION and 1 <= height <= MAX_IMAGE_DIMENSION):
                    raise ValueError(f"image dimensions must not exceed {MAX_IMAGE_DIMENSION} pixels per side")
                if getattr(picture, "n_frames", 1) != 1:
                    raise ValueError("image must contain a single frame")
                picture.verify()
            # verify() inspects container integrity; load() additionally exercises
            # the decoder so truncated compressed data cannot reach the model.
            with Image.open(BytesIO(data)) as picture:
                picture.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("image bytes are invalid or incomplete") from None
    return ValidatedImage(mime, width, height, data)
