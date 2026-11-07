"""Read-only image diagnostics for dataset preparation, independent of inference.

Two fingerprints answer different questions:
- encoded SHA-256: are the files byte-for-byte identical?
- decoded RGB SHA-256: do they contain exactly the same oriented RGB pixels,
  despite different metadata or lossless encodings?

This is exact duplicate detection, not face recognition or perceptual similarity.
EXIF orientation is applied only to the fingerprint; no image file or production
preprocessing behavior is changed.
"""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from app.services.analysis.errors import PhotoValidationError
from app.services.analysis.tracing import content_hash
from app.utils.image_utils import validate_image


class ImageInspection(BaseModel):
    encoded_sha256: str | None = None
    rgb_sha256: str | None = None
    width: int | None = None
    height: int | None = None
    issues: list[str] = Field(default_factory=list)


def inspect_image_path(path: Path) -> ImageInspection:
    """Force decoding so truncated files fail during preparation, not inference."""
    inspection = ImageInspection()
    try:
        image_bytes = path.read_bytes()
    except FileNotFoundError:
        inspection.issues.append("IMAGE_MISSING")
        return inspection
    except OSError:
        inspection.issues.append("IMAGE_READ_FAILED")
        return inspection

    inspection.encoded_sha256 = content_hash(image_bytes)
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            with ImageOps.exif_transpose(source).convert("RGB") as oriented:
                inspection.width, inspection.height = oriented.size
                # Dimensions are part of identity: different row layouts are not
                # the same image, even when their byte sequences happen to match.
                identity = f"RGB:{oriented.width}x{oriented.height}:".encode()
                inspection.rgb_sha256 = content_hash(identity + oriented.tobytes())
        # Reuse the application's existing hard resolution/format rules.
        # Soft lighting/detail heuristics are not a dataset acceptance threshold.
        validate_image(image_bytes).close()
    except PhotoValidationError as exc:
        inspection.issues.append(exc.error_code)
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        inspection.issues.append("PHOTO_UNREADABLE")
    return inspection
