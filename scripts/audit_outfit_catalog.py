"""Audit outfitseed.csv vs SQLite outfits table."""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from urllib.parse import urlparse

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import SessionLocal, init_db
from app.db.models import Outfit

CSV_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "outfitseed.csv")
)
VALID_OCCASIONS = {"daily", "work", "party", "date"}
VALID_CATEGORIES = {"top", "bottom", "set", "outerwear", "accessory", "footwear"}


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _safe_console_text(value: object) -> str:
    text = str(value)
    return text.encode("ascii", errors="replace").decode("ascii")


def load_csv_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _clean(row.get(key)) for key in row})
    return rows


def _in_csv(*, name: str, url: str, csv_urls: set[str], csv_names: set[str]) -> bool:
    return (bool(url) and url in csv_urls) or (bool(name) and name in csv_names)


def check_image_url(url: str, timeout: int = 8) -> tuple[str, int | None]:
    if not url:
        return ("missing", None)
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        status = response.status_code
        response.close()
        if status == 200:
            return ("ok", status)
        return ("http_error", status)
    except Exception as exc:
        return (f"error:{type(exc).__name__}", None)


def main() -> None:
    csv_rows = load_csv_rows()
    csv_urls = {row["shopee_url"] for row in csv_rows if row.get("shopee_url")}
    csv_names = {row["name"] for row in csv_rows if row.get("name")}

    print("=== CSV audit ===")
    print(f"path: {CSV_PATH}")
    print(f"rows: {len(csv_rows)}")

    required = [
        "name",
        "category",
        "image_url",
        "shopee_url",
        "price_range",
        "occasion",
    ]
    missing_required = []
    for index, row in enumerate(csv_rows, start=2):
        missing = [field for field in required if not row.get(field)]
        if missing:
            missing_required.append((index, missing))
    print(f"missing_required_rows: {len(missing_required)}")

    dup_names = [
        name
        for name, count in Counter(
            row["name"] for row in csv_rows if row["name"]
        ).items()
        if count > 1
    ]
    dup_urls = [
        url
        for url, count in Counter(
            row["shopee_url"] for row in csv_rows if row["shopee_url"]
        ).items()
        if count > 1
    ]
    print(f"duplicate_names: {len(dup_names)}")
    print(f"duplicate_shopee_urls: {len(dup_urls)}")

    bad_categories = sorted(
        {
            row["category"]
            for row in csv_rows
            if row.get("category") not in VALID_CATEGORIES
        }
    )
    bad_occasions = sorted(
        {
            row["occasion"]
            for row in csv_rows
            if row.get("occasion") not in VALID_OCCASIONS
        }
    )
    print(f"invalid_categories: {bad_categories or 'none'}")
    print(f"invalid_occasions: {bad_occasions or 'none'}")

    genders = Counter(row.get("gender") or "(empty)" for row in csv_rows)
    print(f"gender_values: {dict(genders)}")

    hosts = Counter(urlparse(row.get("image_url") or "").netloc for row in csv_rows)
    print(f"image_hosts: {dict(hosts)}")

    zara_rows = [
        row for row in csv_rows if "zara.net" in (row.get("image_url") or "").lower()
    ]
    print(f"zara_image_rows: {len(zara_rows)}")

    missing_season_bias = [
        row["name"] for row in csv_rows if not row.get("season_bias")
    ]
    missing_body_shape_bias = [
        row["name"] for row in csv_rows if not row.get("body_shape_bias")
    ]
    print(f"rows_missing_season_bias: {len(missing_season_bias)}")
    print(f"rows_missing_body_shape_bias: {len(missing_body_shape_bias)}")

    occasion_dist = Counter(row.get("occasion") for row in csv_rows)
    category_dist = Counter(row.get("category") for row in csv_rows)
    print(f"occasion_distribution: {dict(occasion_dist)}")
    print(f"category_distribution: {dict(category_dist)}")

    print("\n=== DB audit ===")
    init_db()
    db = SessionLocal()
    try:
        outfits = db.query(Outfit).all()
        print(f"db_outfits: {len(outfits)}")

        extras = [
            outfit
            for outfit in outfits
            if not _in_csv(
                name=outfit.name or "",
                url=outfit.shopee_url or "",
                csv_urls=csv_urls,
                csv_names=csv_names,
            )
        ]
        missing_in_db = []
        db_by_url = {o.shopee_url: o for o in outfits if o.shopee_url}
        db_by_name = {o.name: o for o in outfits if o.name}
        for row in csv_rows:
            existing = (
                db_by_url.get(row["shopee_url"]) if row.get("shopee_url") else None
            )
            if existing is None:
                existing = db_by_name.get(row.get("name"))
            if existing is None:
                missing_in_db.append(row["name"])

        with_vector = [o for o in outfits if o.vector_blob]
        without_vector = [o for o in outfits if not o.vector_blob]
        backends = Counter(o.embedding_backend or "(none)" for o in outfits)
        models = Counter(o.embedding_model or "(none)" for o in outfits)
        male_count = sum(
            1
            for o in outfits
            if (o.gender or "").lower() in {"male", "m", "nam", "men", "man"}
        )

        print(f"db_extras_not_in_csv: {len(extras)}")
        print(f"csv_rows_missing_in_db: {len(missing_in_db)}")
        print(f"with_vector_blob: {len(with_vector)}")
        print(f"without_vector_blob: {len(without_vector)}")
        print(f"embedding_backend: {dict(backends)}")
        print(f"embedding_model: {dict(models)}")
        print(f"gender_male_rows: {male_count}")

        if without_vector:
            print("\nOutfits missing vector:")
            for outfit in without_vector[:10]:
                print(f"  - id={outfit.id} | {_safe_console_text(outfit.name)}")

        if extras:
            print("\nDB extras (not in CSV):")
            for outfit in extras[:10]:
                print(f"  - id={outfit.id} | {_safe_console_text(outfit.name)}")

    finally:
        db.close()

    print("\n=== Image URL spot-check (slow) ===")
    targets = (
        zara_rows[:5]
        + [
            row
            for row in csv_rows
            if "zara.net" not in (row.get("image_url") or "").lower()
        ][:3]
    )
    for row in targets:
        status, code = check_image_url(row.get("image_url", ""))
        print(f"  [{status} {code}] {_safe_console_text(row['name'][:60])}")


if __name__ == "__main__":
    main()
