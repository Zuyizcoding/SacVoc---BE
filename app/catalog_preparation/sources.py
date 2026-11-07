"""Audit source bytes before they enter the embedding pipeline.

Intuition: an HTTP 200 can contain an error page, and different URLs can contain
the same image. We therefore decode every unique URL, retain the original bytes,
and record both file and decoded-RGB hashes. This checks technical availability,
not product accuracy, ownership, or permission to redistribute the image.

Schema: one ImageSource per distinct URL; row numbers refer to CSV data rows
(starting at 1, excluding the header). Pipeline: GET -> size limit -> decode RGB
-> hashes -> local cache. Failures remain explicit and are never substituted.
"""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 12 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class ImageSource:
    url: str
    csv_rows: list[int]
    status: str
    http_status: int | None = None
    final_url: str | None = None
    content_type: str | None = None
    byte_count: int = 0
    width: int | None = None
    height: int | None = None
    file_sha256: str | None = None
    rgb_sha256: str | None = None
    cache_file: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_source(url: str, csv_rows: list[int], cache: Path) -> ImageSource:
    """Fetch once using normal TLS verification; do not bypass HTTP denials."""
    details = {"url": url, "csv_rows": csv_rows}
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Expected an absolute HTTPS image URL")
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            details.update(
                http_status=response.status,
                final_url=response.url,
                content_type=response.headers.get("Content-Type"),
            )
            data = response.read(MAX_IMAGE_BYTES + 1)
        details["byte_count"] = len(data)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the 12 MiB download limit")
        # Decode failures are data errors, even when Pillow raises OSError.
        try:
            # verify() checks structure; reopen before loading pixels.
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                rgb = image.convert("RGB")
                rgb.load()
                details.update(width=rgb.width, height=rgb.height)
                pixels = f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Image decode failed: {exc}") from exc
    except HTTPError as exc:
        return ImageSource(
            **details, status="http_error", http_status=exc.code, error=str(exc)
        )
    except (URLError, TimeoutError, OSError) as exc:
        return ImageSource(**details, status="network_error", error=str(exc))
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        return ImageSource(**details, status="invalid_image", error=str(exc))

    # Disk errors must fail the run, not be mislabeled as unavailable URLs.
    digest = hashlib.sha256(data).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    filename = f"{digest}.image"
    (cache / filename).write_bytes(data)
    return ImageSource(
        **details,
        status="ready",
        file_sha256=digest,
        rgb_sha256=hashlib.sha256(pixels).hexdigest(),
        cache_file=filename,
    )


def audit_sources(rows: list[dict[str, str]], cache: Path) -> list[ImageSource]:
    """Bound concurrency to four requests and preserve first-appearance order."""
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows, start=1):
        grouped.setdefault(row["image_url"].strip(), []).append(index)
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = [
            executor.submit(inspect_source, url, indices, cache)
            for url, indices in grouped.items()
        ]
        return [job.result() for job in jobs]
