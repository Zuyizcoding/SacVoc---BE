"""Prune outfits in DB that are not present in outfitseed CSV.

Use case:
- You seeded placeholder outfits earlier (scripts/seed_outfits.py) and later
  seeded from outfitseed.csv (scripts/seed_outfits_csv.py).
- Because CSV seeding is idempotent upsert (doesn't delete), DB can contain
  extra rows not present in the CSV.
- This script deletes those extra rows so DB matches the CSV exactly.

Run:
    cd src/backend
    python -m scripts.prune_outfits_not_in_csv --dry-run
    python -m scripts.prune_outfits_not_in_csv

Options:
    --csv <path>        CSV path (default: src/backend/outfitseed.csv)
    --dry-run           Print what would be deleted, but rollback

Matching rule:
- A DB outfit is considered "in CSV" if its shopee_url matches a CSV shopee_url,
  OR its name matches a CSV name.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import SessionLocal, init_db
from app.db.models import Outfit


def _default_csv_path() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outfitseed.csv"))


def _read_csv_keys(csv_path: str) -> tuple[set[str], set[str], int]:
    urls: set[str] = set()
    names: set[str] = set()
    row_count = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            name = (row.get("name") or "").strip()
            url = (row.get("shopee_url") or "").strip()
            if name:
                names.add(name)
            if url:
                urls.add(url)

    return urls, names, row_count


def _in_csv(*, name: str, url: str, csv_urls: set[str], csv_names: set[str]) -> bool:
    name = (name or "").strip()
    url = (url or "").strip()
    return (bool(url) and url in csv_urls) or (bool(name) and name in csv_names)


def prune(csv_path: str, *, commit: bool) -> None:
    init_db()
    db = SessionLocal()

    try:
        csv_urls, csv_names, csv_rows = _read_csv_keys(csv_path)

        outfits = db.query(Outfit).all()
        extras: list[Outfit] = [
            o
            for o in outfits
            if not _in_csv(
                name=getattr(o, "name", "") or "",
                url=getattr(o, "shopee_url", "") or "",
                csv_urls=csv_urls,
                csv_names=csv_names,
            )
        ]

        print(f"CSV_ROWS={csv_rows}")
        print(f"DB_OUTFITS={len(outfits)}")
        print(f"EXTRA_TO_DELETE={len(extras)}")

        for o in extras[:30]:
            print(f"  DELETE id={o.id} name={o.name!r} url={o.shopee_url!r}")

        if not extras:
            print("OK: DB already matches CSV; nothing to delete")
            return

        for o in extras:
            db.delete(o)

        if commit:
            db.commit()
            print(f"OK: Deleted {len(extras)} extra outfit(s)")
        else:
            db.rollback()
            print("DRY-RUN: no changes committed")

    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete outfits in DB that are not present in the CSV")
    parser.add_argument("--csv", dest="csv_path", default=_default_csv_path())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv_path)
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    prune(csv_path, commit=not args.dry_run)


if __name__ == "__main__":
    main()
