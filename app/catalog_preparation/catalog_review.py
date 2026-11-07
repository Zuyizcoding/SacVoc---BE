"""Audit a frozen real catalog and render an offline coordinator gallery.

Intuition: a decodable image can still be the wrong product. Structural checks
flag missing data and shared-image disagreements for humans, without claiming
visual truth or changing the seed. A shared URL with different metadata is a
review warning, not automatic proof that a row is wrong.

Schema: typed issues include code, severity, affected CSV data-row IDs, fields
and explanation. Pipeline: verify bundle -> read CSV/DB -> find issues -> copy
cached images -> render one card per row -> write evidence. No app DB startup,
AI calls, downloads or changes to the catalog are involved.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Literal

from app.data.body_shapes import MALE_BODY_SHAPES
from app.evaluation.sqlite_catalog import readonly_catalog
from app.services.recommendation.query_builder import SEASON_KEYWORDS

from .encoder import file_sha256
from .integrity import verify_manifest
from .review_assets import (
    copy_local_images,
    external_link,
    image_html,
    load_sources,
    review_document,
)
from .workflow import read_seed_rows, write_json

VALID_CATEGORIES = {"top", "bottom", "set", "outerwear", "accessory", "footwear"}
VALID_OCCASIONS = {"daily", "work", "party", "date"}
REQUIRED_FIELDS = (
    "name",
    "category",
    "image_url",
    "shopee_url",
    "price_range",
    "occasion",
)
INFERRED_FIELDS = (
    "fit",
    "formality",
    "fabric",
    "color_family",
    "season_bias",
    "body_shape_bias",
)
CONFLICT_FIELDS = ("name", "category", "shopee_url", "price_range")


@dataclass(frozen=True)
class CatalogIssue:
    code: str
    severity: Literal["error", "warning", "info"]
    row_ids: list[int]
    fields: list[str]
    explanation: str


def inspect_catalog(
    rows: list[dict], csv_rows: list[dict], sources: list
) -> list[CatalogIssue]:
    """Pure structural checks. Keep every disagreement visible and actionable."""
    issues = []
    if len(rows) != len(csv_rows):
        issues.append(
            CatalogIssue(
                "row_count_mismatch", "error", [], [], "CSV and DB row counts differ"
            )
        )
    source_by_url = {source.url: source for source in sources}
    grouped = defaultdict(list)
    for row in rows:
        row_id = row["id"]
        grouped[row["image_url"]].append(row)
        missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
        if missing:
            issues.append(
                CatalogIssue(
                    "missing_required",
                    "error",
                    [row_id],
                    missing,
                    "Required catalog fields are blank",
                )
            )
        inferred = [field for field in INFERRED_FIELDS if not row.get(field)]
        if inferred:
            issues.append(
                CatalogIssue(
                    "missing_metadata",
                    "warning",
                    [row_id],
                    inferred,
                    "Ranking metadata is missing; no values were invented",
                )
            )
        invalid = []
        if row["category"] not in VALID_CATEGORIES:
            invalid.append("category")
        if row["occasion"] not in VALID_OCCASIONS:
            invalid.append("occasion")
        if row["gender"] not in {"male", "female", "any"}:
            invalid.append("gender")
        for field, allowed in (
            ("body_shape_bias", set(MALE_BODY_SHAPES)),
            (
                "season_bias",
                set(SEASON_KEYWORDS) | {"Spring", "Summer", "Autumn", "Winter"},
            ),
        ):
            if any(
                tag.strip() not in allowed
                for tag in (row.get(field) or "").split("|")
                if tag.strip()
            ):
                invalid.append(field)
        if invalid:
            issues.append(
                CatalogIssue(
                    "unknown_taxonomy",
                    "error",
                    [row_id],
                    invalid,
                    "Stored values are outside the declared taxonomy",
                )
            )
        if 1 <= row_id <= len(csv_rows):
            original = csv_rows[row_id - 1]
            changed = [
                field
                for field in ("name", "image_url", "shopee_url", "price_range")
                if row[field] != original[field]
            ]
            if changed:
                issues.append(
                    CatalogIssue(
                        "csv_identity_mismatch",
                        "error",
                        [row_id],
                        changed,
                        "Stable CSV identity fields differ from the seeded DB",
                    )
                )
        else:
            issues.append(
                CatalogIssue(
                    "unknown_csv_row",
                    "error",
                    [row_id],
                    [],
                    "DB ID does not identify a CSV data row",
                )
            )
        source = source_by_url.get(row["image_url"])
        if source is None or row_id not in source.csv_rows:
            issues.append(
                CatalogIssue(
                    "source_mapping_mismatch",
                    "error",
                    [row_id],
                    ["image_url"],
                    "Source audit does not map this row to its image",
                )
            )
        elif source.status != "ready":
            issues.append(
                CatalogIssue(
                    "image_unavailable",
                    "warning",
                    [row_id],
                    ["image_url"],
                    f"Frozen source outcome: {source.status}, HTTP {source.http_status}",
                )
            )
        if (row["vector_blob"] is not None) != (
            source is not None and source.status == "ready"
        ):
            issues.append(
                CatalogIssue(
                    "encoding_coverage_mismatch",
                    "error",
                    [row_id],
                    ["vector_blob"],
                    "Encoded state and source availability disagree",
                )
            )
    for url, group in grouped.items():
        if len(group) < 2:
            continue
        ids = [row["id"] for row in group]
        issues.append(
            CatalogIssue(
                "shared_image_url",
                "info",
                ids,
                ["image_url"],
                "Multiple rows reference the same image; retained by design",
            )
        )
        conflicts = [
            field
            for field in CONFLICT_FIELDS
            if len({row.get(field) for row in group}) > 1
        ]
        if conflicts:
            issues.append(
                CatalogIssue(
                    "shared_image_metadata_conflict",
                    "warning",
                    ids,
                    conflicts,
                    "Same image, different product metadata: review before correcting anything",
                )
            )
    return issues


def render_gallery(
    rows: list[dict], issues: list[CatalogIssue], images: dict[str, str]
) -> str:
    cards = []
    for row in rows:
        flags = [issue for issue in issues if row["id"] in issue.row_ids]
        warnings = "".join(
            f'<p class="flag">{escape(issue.code)}: {escape(issue.explanation)}</p>'
            for issue in flags
        )
        metadata = " | ".join(
            f"{field}: {row.get(field) or '(blank)'}"
            for field in (
                "category",
                "occasion",
                "gender",
                "price_range",
                *INFERRED_FIELDS,
            )
        )
        cards.append(
            f'<article id="row-{row["id"]}"><h2>Row {row["id"]}: {escape(row["name"])}</h2>'
            + image_html(images.get(row["image_url"]), row["name"])
            + f'<p class="meta">{escape(metadata)}</p>'
            + external_link(row["shopee_url"], "Product source")
            + warnings
            + "</article>"
        )
    counts = Counter(issue.severity for issue in issues)
    introduction = (
        f"{len(rows)} CSV rows. {counts['error']} structural errors, "
        f"{counts['warning']} review warnings. Coordinator only: this page shows "
        "private ranking metadata. Do not give it to independent raters."
    )
    return review_document(
        "Catalog review — frozen v2 bundle",
        introduction,
        '<div class="grid">' + "".join(cards) + "</div>",
    )


def prepare_catalog_review(bundle: Path, csv_path: Path, output: Path) -> dict:
    integrity = verify_manifest(bundle)
    csv_rows = read_seed_rows(csv_path)
    with readonly_catalog(bundle / "catalog.db") as connection:
        rows = [
            dict(row) for row in connection.execute("SELECT * FROM outfits ORDER BY id")
        ]
    issues = inspect_catalog(rows, csv_rows, load_sources(bundle))
    output.mkdir(parents=True, exist_ok=False)
    images = copy_local_images(bundle, output)
    (output / "gallery.html").write_text(
        render_gallery(rows, issues, images), encoding="utf-8"
    )
    report = {
        "row_count": len(rows),
        "encoded_rows": sum(row["vector_blob"] is not None for row in rows),
        "local_image_count": len(set(images.values())),
        "issue_counts": dict(Counter(issue.code for issue in issues)),
        "severity_counts": dict(Counter(issue.severity for issue in issues)),
        "issues": [asdict(issue) for issue in issues],
        "integrity": integrity,
        "csv_sha256": file_sha256(csv_path),
        "interpretation": "Structural audit and coordinator visual-review aid; no automatic corrections.",
    }
    write_json(output / "report.json", report)
    verify_manifest(bundle)
    return report
