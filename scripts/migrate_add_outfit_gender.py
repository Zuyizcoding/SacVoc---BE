"""One-off migration: add `gender` column to `outfits` table (SQLite).

Normally you don't need to run this manually because `init_db()` applies a
lightweight migration automatically. This script is here as a safe explicit
command to run on an existing DB.

Run:
    cd src/backend
    python -m scripts.migrate_add_outfit_gender
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text

from app.db.database import engine, init_db


def main() -> None:
    init_db()

    with engine.begin() as conn:
        cols = conn.execute(text("PRAGMA table_info(outfits)")).fetchall()
        col_names = {row[1] for row in cols}
        if "gender" in col_names:
            print("OK Migration skipped: outfits.gender already exists")
            return

        conn.execute(
            text(
                "ALTER TABLE outfits "
                "ADD COLUMN gender VARCHAR(10) NOT NULL DEFAULT 'any'"
            )
        )
        print("OK Migration applied: added outfits.gender")


if __name__ == "__main__":
    main()
