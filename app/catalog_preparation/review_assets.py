"""Shared offline HTML/image utilities for catalog and reviewer artifacts.

Intuition: a remote URL that worked yesterday may fail during review. Copy the
already audited bytes into each standalone folder, validating checksums first.
No image is regenerated, relabeled or fetched from the network.

Schema: URL -> relative local image path. Pipeline: read source audit -> confine
cache paths to images/ -> verify original byte hash -> copy supported bytes.
All dynamic HTML text is escaped. The page policy forbids scripts and automatic
external requests; explicit HTTPS links can still be opened by the reviewer.
"""

import json
import shutil
from html import escape
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from .encoder import file_sha256
from .sources import ImageSource

IMAGE_SUFFIXES = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}
PAGE_STYLE = """
body{font:16px/1.5 system-ui,sans-serif;background:#f5f5f2;color:#172b29;margin:0}
header,main{max-width:1250px;margin:auto;padding:24px}
header{background:#163e37;color:white;border-radius:0 0 18px 18px}
header p{max-width:950px}h1{margin:0;font-size:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px}
article{background:white;border:1px solid #d6dfda;border-radius:12px;padding:18px}
h2{font-size:18px;margin:8px 0}img{width:100%;height:280px;object-fit:contain;background:#f1f1ed}
.meta{font-size:14px;color:#42554e}.flag{background:#fff0cc;border-left:4px solid #ae7017;padding:8px}
.missing{height:220px;display:grid;place-items:center;background:#f7e6e4}
nav a{margin-right:16px}a{color:#146751}code{overflow-wrap:anywhere}
"""


def load_sources(bundle: Path) -> list[ImageSource]:
    payload = json.loads((bundle / "sources.json").read_text(encoding="utf-8"))
    sources = [ImageSource(**item) for item in payload["sources"]]
    if len({source.url for source in sources}) != len(sources):
        raise ValueError("Source audit contains duplicate URL entries")
    return sources


def cached_image_path(bundle: Path, source: ImageSource) -> Path:
    if source.status != "ready" or not source.cache_file or not source.file_sha256:
        raise ValueError("A ready source needs a cache filename and checksum")
    root = (bundle / "images").resolve(strict=True)
    path = (root / source.cache_file).resolve(strict=True)
    if path.parent != root:
        raise ValueError("Cached image path escapes the bundle image directory")
    if file_sha256(path) != source.file_sha256:
        raise ValueError(f"Cached image checksum mismatch: {source.cache_file}")
    return path


def copy_local_images(bundle: Path, output: Path) -> dict[str, str]:
    """Copy original bytes, deduplicated by hash; return portable relative paths."""
    images = output / "images"
    images.mkdir(parents=True, exist_ok=False)
    paths = {}
    for source in load_sources(bundle):
        if source.status != "ready":
            continue
        original = cached_image_path(bundle, source)
        with Image.open(original) as image:
            suffix = IMAGE_SUFFIXES.get(image.format)
        if suffix is None:
            raise ValueError(
                f"Unsupported offline-browser image format: {original.name}"
            )
        filename = source.file_sha256 + suffix
        target = images / filename
        if not target.exists():
            shutil.copyfile(original, target)
        paths[source.url] = "images/" + filename
    return paths


def external_link(url: str, label: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        return escape(label) + " (no HTTPS link)"
    return f'<a href="{escape(url, quote=True)}" rel="noreferrer">{escape(label)}</a>'


def image_html(path: str | None, label: str) -> str:
    if path is None:
        return '<div class="missing">Image unavailable in this frozen bundle</div>'
    return f'<img src="{escape(path, quote=True)}" alt="{escape(label, quote=True)}" loading="lazy">'


def review_document(title: str, introduction: str, content: str) -> str:
    """Content is markup assembled by our escaped renderers, never raw CSV HTML."""
    # Explicit file: supports local pages with opaque browser origins.
    policy = "default-src 'none'; img-src 'self' file:; style-src 'unsafe-inline'; base-uri 'none'"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{escape(policy, quote=True)}">'
        f"<title>{escape(title)}</title><style>{PAGE_STYLE}</style></head><body>"
        f"<header><h1>{escape(title)}</h1><p>{escape(introduction)}</p></header>"
        f"<main>{content}</main></body></html>"
    )
