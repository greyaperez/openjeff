import base64
from io import BytesIO

from PIL import Image
import pytest

from djev.images import validate_image_data_url


def data_url(size=(32, 32), format="PNG", mime="image/png"):
    image = BytesIO()
    Image.new("RGB", size, "blue").save(image, format=format)
    return f"data:{mime};base64," + base64.b64encode(image.getvalue()).decode()


def test_decoded_image_has_verified_dimensions_and_mime():
    result = validate_image_data_url(data_url())
    assert (result.width, result.height, result.mime_type) == (32, 32, "image/png")
    assert "data=" not in repr(result)


@pytest.mark.parametrize("value", [
    "https://example.com/image.png", "file:///etc/passwd", "data:image/svg+xml;base64,PHN2Zy8+",
    "data:image/png;base64,!invalid!", "data:image/png;base64,",
    data_url(size=(2049, 1)), data_url(mime="image/jpeg"),
])
def test_invalid_image_never_needs_network_or_decoder_fallback(value):
    with pytest.raises(ValueError):
        validate_image_data_url(value)


def test_animated_png_is_rejected():
    stream = BytesIO()
    first = Image.new("RGB", (4, 4), "red")
    second = Image.new("RGB", (4, 4), "blue")
    first.save(stream, format="PNG", save_all=True, append_images=[second], duration=100)
    with pytest.raises(ValueError, match="single frame"):
        validate_image_data_url("data:image/png;base64," + base64.b64encode(stream.getvalue()).decode())


def test_oversized_base64_is_rejected_before_image_decode(monkeypatch):
    import djev.images
    monkeypatch.setattr(djev.images, "MAX_IMAGE_BYTES", 3)
    with pytest.raises(ValueError, match="bytes"):
        validate_image_data_url(data_url())
