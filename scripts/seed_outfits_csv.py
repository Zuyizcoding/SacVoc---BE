"""Seed outfits from `outfitseed.csv` into DB.

Requirements (per spec):
- Read CSV as UTF-8 (supports BOM via utf-8-sig)
- Map columns: name, category, image_url, shopee_url, price_range, occasion
- Idempotent: upsert by `shopee_url` (preferred) or `name`

Run:
    cd src/backend
    python -m scripts.seed_outfits_csv

Optional:
    python -m scripts.seed_outfits_csv --csv outfitseed.csv
"""

import argparse
import csv
import os
import re
import sys
from typing import Any

# Thêm src/backend vào path để import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import SessionLocal, init_db
from app.db.models import Outfit


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_gender(value: Any) -> str:
    v = _clean(value).lower()
    if not v:
        return ""
    if v in {"any", "all", "unisex"}:
        return "any"
    if v in {"male", "m", "nam", "men", "man"}:
        return "male"
    if v in {"female", "f", "nu", "nữ", "women", "woman"}:
        return "female"
    # Allow already-normalized/custom values, but keep it trimmed.
    return v


def _normalize_bias_list(value: Any) -> str:
    raw = _clean(value)
    if not raw:
        return ""
    parts = re.split(r"[|,;/]+", raw)
    normalized = [part.strip() for part in parts if part.strip()]
    deduped = list(dict.fromkeys(normalized))
    return "|".join(deduped)


def _extract_seed_biases(name: str) -> tuple[str, str]:
    match = re.search(r"\(([^-]+)-\s*([^)]+)\)", name)
    if not match:
        return ("", "")
    season = match.group(1).strip()
    shape = match.group(2).strip()
    return (season, shape)


def _infer_fit(name: str, category: str) -> str:
    text = f"{name} {category}".lower()
    if any(token in text for token in ("slimfit", "dáng ôm", "ôm", "fitted", "tailored")):
        return "tailored" if category in {"set", "outerwear"} else "slim"
    if any(token in text for token in ("form rộng", "oversize", "ống suông", "boxy", "wide", "rộng rãi")):
        return "relaxed"
    if category == "set" and any(token in text for token in ("suit", "vest", "blazer")):
        return "tailored"
    return "regular"


def _infer_fabric(name: str) -> str:
    text = name.lower()
    if any(token in text for token in ("denim", "jeans", "jean")):
        return "denim"
    if any(token in text for token in ("lụa", "silk")):
        return "silk"
    if any(token in text for token in ("len", "knit", "cardigan")):
        return "knit"
    if any(token in text for token in ("da ", "biker", "leather")):
        return "leather"
    if any(token in text for token in ("đũi", "linen")):
        return "linen"
    if any(token in text for token in ("nhung", "corduroy")):
        return "corduroy"
    if any(token in text for token in ("kaki", "cargo", "chinos")):
        return "cotton_twill"
    if any(token in text for token in ("nỉ", "hoodie", "sweat")):
        return "fleece"
    if any(token in text for token in ("vest", "suit", "blazer", "âu", "tây")):
        return "woven"
    return "cotton"


def _infer_formality(name: str, occasion: str) -> str:
    text = f"{name} {occasion}".lower()
    if any(token in text for token in ("suit", "vest", "blazer", "formal")):
        return "formal" if occasion in {"party", "date"} else "business_casual"
    if occasion == "work":
        return "business_casual"
    if occasion in {"party", "date"}:
        return "smart_casual"
    return "casual"


def _infer_color_family(name: str, season: str) -> str:
    text = name.lower()
    if any(token in text for token in ("olive", "rêu", "nâu", "be", "kem", "camel", "đất", "terracotta")):
        return "earth"
    if any(token in text for token in ("navy", "xanh", "mint", "baby", "ghi", "xám", "đen", "trắng")):
        return "cool"
    if any(token in text for token in ("burgundy", "đỏ rượu", "đỏ", "wine")):
        return "rich"
    lowered_season = season.lower()
    if "autumn" in lowered_season:
        return "earth"
    if "summer" in lowered_season:
        return "soft_cool"
    if "winter" in lowered_season:
        return "high_contrast"
    if "spring" in lowered_season:
        return "warm_bright"
    return "neutral"


def _infer_contrast_level(season: str) -> str:
    lowered = season.lower()
    if "winter" in lowered:
        return "high"
    if "summer" in lowered:
        return "low"
    if "spring" in lowered:
        return "medium_high"
    if "autumn" in lowered:
        return "medium"
    return "medium"


def _infer_silhouette(category: str, shape: str) -> str:
    normalized = shape.strip().lower()
    if normalized == "inverted triangle":
        return "balanced lower body"
    if normalized == "triangle":
        return "clean shoulder line"
    if normalized == "oval":
        return "vertical line"
    if normalized == "rectangle":
        return "layered structure"
    if normalized == "trapezoid":
        return "tailored balance"
    if category == "outerwear":
        return "structured outer layer"
    if category == "bottom":
        return "straight leg"
    return "clean menswear line"


def _build_metadata_payload(row: dict[str, Any], *, name: str, category: str, occasion: str) -> dict[str, str]:
    season_bias, body_shape_bias = _extract_seed_biases(name)
    fit = _clean(row.get("fit")) or _infer_fit(name, category)
    silhouette = _clean(row.get("silhouette")) or _infer_silhouette(category, body_shape_bias)
    formality = _clean(row.get("formality")) or _infer_formality(name, occasion)
    fabric = _clean(row.get("fabric")) or _infer_fabric(name)
    color_family = _clean(row.get("color_family")) or _infer_color_family(name, season_bias)
    contrast_level = _clean(row.get("contrast_level")) or _infer_contrast_level(season_bias)

    raw_shape_bias = _normalize_bias_list(row.get("body_shape_bias")) or body_shape_bias
    raw_season_bias = _normalize_bias_list(row.get("season_bias")) or season_bias

    return {
        "fit": fit,
        "silhouette": silhouette,
        "formality": formality,
        "fabric": fabric,
        "color_family": color_family,
        "contrast_level": contrast_level,
        "body_shape_bias": raw_shape_bias,
        "season_bias": raw_season_bias,
    }


def _apply_updates(outfit: Outfit, payload: dict[str, str]) -> bool:
    """Apply non-empty payload values onto outfit. Returns True if changed."""
    changed = False
    for key, value in payload.items():
        if not hasattr(outfit, key):
            continue
        if value == "":
            continue
        if getattr(outfit, key) != value:
            setattr(outfit, key, value)
            changed = True
    return changed


def seed(csv_path: str, *, commit: bool = True) -> None:
    init_db()
    db = SessionLocal()

    inserted = 0
    updated = 0
    skipped = 0

    try:
        existing_outfits = db.query(Outfit).all()
        by_url: dict[str, Outfit] = {}
        by_name: dict[str, Outfit] = {}

        for outfit in existing_outfits:
            url_key = _clean(getattr(outfit, "shopee_url", ""))
            name_key = _clean(getattr(outfit, "name", ""))
            if url_key and url_key not in by_url:
                by_url[url_key] = outfit
            if name_key and name_key not in by_name:
                by_name[name_key] = outfit

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            required = {
                "name",
                "category",
                "image_url",
                "shopee_url",
                "price_range",
                "occasion",
            }
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"CSV headers must include {sorted(required)}; got {reader.fieldnames}"
                )

            for row in reader:
                name = _clean(row.get("name"))
                if not name:
                    continue

                payload = {
                    "name": name,
                    "category": _clean(row.get("category")),
                    "image_url": _clean(row.get("image_url")),
                    "shopee_url": _clean(row.get("shopee_url")),
                    "price_range": _clean(row.get("price_range")),
                    "occasion": _clean(row.get("occasion")),
                }
                payload.update(
                    _build_metadata_payload(
                        row,
                        name=name,
                        category=payload["category"],
                        occasion=payload["occasion"],
                    )
                )

                # Optional column: gender (any|male|female). If absent/empty → keep default.
                gender = _normalize_gender(row.get("gender"))
                if gender:
                    payload["gender"] = gender

                lookup_url = payload["shopee_url"]
                existing = by_url.get(lookup_url) if lookup_url else None
                if existing is None:
                    existing = by_name.get(name)

                if existing is None:
                    outfit = Outfit(**payload)
                    db.add(outfit)
                    inserted += 1

                    if lookup_url:
                        by_url.setdefault(lookup_url, outfit)
                    by_name.setdefault(name, outfit)
                    continue

                changed = _apply_updates(existing, payload)

                if changed:
                    updated += 1
                else:
                    skipped += 1

                # Keep lookup maps in sync in case name/url changed.
                current_name = _clean(getattr(existing, "name", ""))
                current_url = _clean(getattr(existing, "shopee_url", ""))
                if current_name:
                    by_name.setdefault(current_name, existing)
                if current_url:
                    by_url.setdefault(current_url, existing)

        if commit:
            db.commit()
        else:
            db.rollback()

        print(
            f"OK Seed outfits CSV done: inserted={inserted}, updated={updated}, skipped={skipped}"
        )

        if not commit:
            print("DRY-RUN: no changes committed")

        if inserted + updated == 0:
            print("   (No changes applied)")

    finally:
        db.close()


def _default_csv_path() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outfitseed.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed outfits from CSV into DB")
    parser.add_argument("--csv", dest="csv_path", default=_default_csv_path())
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and compute changes but do not commit to DB",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv_path)
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    seed(csv_path, commit=not args.dry_run)


if __name__ == "__main__":
    main()
