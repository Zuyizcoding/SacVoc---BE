"""Storage-independent catalog rules shared by SQL loading and offline snapshots."""

from collections.abc import Sequence
from typing import TypeVar

from .types import RankableOutfit

MALE_GENDER_ALIASES = ("male", "m", "nam", "man", "men")
OutfitRecord = TypeVar("OutfitRecord", bound=RankableOutfit)


def unique_image_outfits(outfits: Sequence[OutfitRecord]) -> list[OutfitRecord]:
    """Keep the first record per non-empty, trimmed image URL.

    Preserve input order: it is also the tie-break order of production ranking.
    Empty URLs retain their historical behavior and are not deduplicated here.
    """
    unique = []
    seen_images: set[str] = set()
    for outfit in outfits:
        image_key = str(outfit.image_url or "").strip()
        if image_key and image_key in seen_images:
            continue
        if image_key:
            seen_images.add(image_key)
        unique.append(outfit)
    return unique
