"""Mọi test dùng SQLite trong bộ nhớ và khóa giả, không chạm dữ liệu demo."""

import os

os.environ["DATABASE_URL"] = "sqlite://"
os.environ.setdefault("GEMINI_API_KEY", "unit-test-key")
