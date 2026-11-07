# F3 API integration, offline catalog review and independent-rating pilot

Read this guide in order: intuition -> schema -> pipeline -> code details.
The artifacts prepare human review; they do not supply relevance grades.
Measured results are in [catalog-review-verification.md](catalog-review-verification.md).

## Intuition

A correct vector space is only one part of retrieval quality. The real HTTP
adapter must still validate requests, serialize results and reject incompatible
data. Images can decode correctly while depicting a different product. Finally,
a recommendation score is not an independent relevance judgment.

This stage therefore keeps three questions separate:

1. Does the existing FastAPI F3 endpoint work with the v2 bundle?
2. Which catalog rows require a coordinator to inspect their source/metadata?
3. Can two reviewers judge the same frozen questions independently and offline?

No model is loaded, no Gemini request is made, and no application setting changes.
All work uses the explicit v2 bundle and fresh output paths.

## Schema and implementation map

| Module | Responsibility |
| --- | --- |
| catalog_preparation/integrity.py | Shared manifest and preserved-baseline verification |
| catalog_preparation/api_verification.py | Parent isolation plus child-only application import |
| catalog_preparation/catalog_review.py | Typed CatalogIssue records and coordinator gallery |
| catalog_preparation/review_assets.py | Safe local-image copies and escaped HTML |
| catalog_preparation/pilot.py | Hypothetical scenario plan, offline forms and reviewer ZIPs |
| evaluation/annotation_packet.py | Existing task freezing, reviewer shuffling and blank CSV |
| evaluation/annotation_csv.py | Existing strict completed-rating validation |

CatalogIssue records code, severity, row_ids, fields and explanation. Row IDs
count CSV data rows starting at 1, excluding the header. Warnings do not authorize
automatic correction. Info-level shared-image groups are retained by design.

The pilot reuses the existing AnnotationPlan/Packet/Task contracts. Its
synthetic=false field refers to the real catalog embedding identity. The profiles
are explicitly hypothetical scenarios, not observed subjects or F1/F2 predictions.
Each subject_id starts with hypothetical-profile-. Every case is dev.

## 1. API integration pipeline

The public CLI verifies bundle hashes, backs up the catalog through SQLite into a
TemporaryDirectory, and starts a separate Python process. That child receives an
explicit DATABASE_URL for the copy and F3_QUERY_VECTORS_PATH for the saved v2 JSON.
Only then does it import app.main, so startup migrations cannot touch a user DB.

The worker inserts one reusable, hypothetical test profile into the copy and
uses FastAPI TestClient to exercise the real routes and real recommendation code.
This is in-process ASGI integration, not a listening server/network benchmark.

Checks include all 240 profiles twice, outfit details, missing F1/F2, invalid
headers/parameters, default daily filtering, readiness, a mixed-version catalog,
query/catalog version mismatch, and an empty catalog. Negative cases retain the
current HTTP/JSON contracts: a not-ready recommendation is HTTP 200 with
success=false and error_code=RECOMMENDATION_NOT_READY.

Gemini inference and runtime encoder fallback are guarded; external TCP is
blocked. After the worker exits, the temporary DB is removed and source hashes
are checked again. No fixture profile survives in the source catalog.

~~~powershell
.venv/verify/Scripts/python.exe -m scripts.verify_bundle_api --bundle build/real-catalog/projected-v2 --output-dir build/review/final/api
~~~

Choose a fresh output path on rerun; existing output directories are refused.

## 2. Structural audit and offline gallery

The audit opens the catalog read-only and compares stable identity fields with
the CSV. It checks required fields, taxonomy, ranking-metadata absence, source-row
mapping, encoding coverage and shared-image product conflicts.

A warning about shared images means the same URL is attached to differing names,
categories, prices or product links. It is evidence for inspection, not proof
that one particular row is wrong. A photo might intentionally show several items.

Cached images are confined to bundle/images/, checked against source SHA-256 and
copied byte-for-byte. No thumbnails or replacement images are generated. The
HTML scales the original bytes, escapes all dynamic text, and has a restrictive
content policy. It contains no script or automatic remote image request.
Explicit HTTPS product links are available for deliberate source inspection.

~~~powershell
.venv/verify/Scripts/python.exe -m scripts.prepare_catalog_review --bundle build/real-catalog/projected-v2 --output-dir build/review/final/catalog
~~~

Open gallery.html locally. This is a COORDINATOR page and includes private ranking
metadata. Do not distribute it to independent raters.

Missing body_shape_bias on footwear may be intentional because shape relevance
is limited. Confirm the convention; do not fabricate a label merely to remove
a warning. The current ranking fallback is preserved.

## 3. Pilot plan and offline reviewer folders

Twenty scenarios cover each of 5 body shapes x 4 occasions exactly once. All 12
seasons appear, with 1 or 2 cases per season. This deterministic design covers
the taxonomy but is not a population sample. There is no held-out test set.

The existing packet builder applies production eligibility filtering and image
deduplication, then includes ALL eligible items for each scenario. It never runs
a ranker to choose a favorable subset. Each reviewer gets a separately shuffled
order of the same task keys.

~~~powershell
.venv/verify/Scripts/python.exe -m scripts.prepare_relevance_pilot --bundle build/real-catalog/projected-v2 --catalog-audit build/review/final/catalog/report.json --output-dir build/review/final/pilot
~~~

The audit must match the bundle fingerprint, and structural errors must be
resolved first. Warnings remain visible in the coordinator summary. Materials
are explicitly draft until the coordinator reviews the data and rubric.

Output:

~~~text
pilot/
  annotation-plan.json
  summary.json
  COORDINATOR.md
  rater-a.zip
  rater-b.zip
  packet/
    private/packet.json
    rater-a/
      ratings.csv
      reference.html
      README.md
      images/
    rater-b/
      ratings.csv
      reference.html
      README.md
      images/
~~~

Each ZIP contains only its assigned reviewer directory, including offline images,
not private/packet.json or the coordinator audit. Extract the ZIP before opening
reference.html. Procedural blinding is not proof of two independent people.

Grades and notes start blank. The allowlisted task context contains season,
shape, occasion, product name/category and image URL; it excludes model scores,
confidence, vectors and catalog bias tags. HTML uses local images but CSV retains
the immutable original task context for downstream validation.

## Coordinator workflow and limits

1. Inspect the two shared-image groups and decide whether mappings are correct.
2. Confirm whether footwear body-shape fields should remain intentionally blank.
3. Confirm or revise the draft 0..3 relevance rubric and uncertainty handling.
4. If source data changes, regenerate the versioned bundle, audit and packet.
   Never patch frozen reviewer context or carry old ratings into changed tasks.
5. After approval, give only rater-a.zip to one person and rater-b.zip to another.
6. Reviewers work independently and edit only grade and notes.
7. Collect both completed UTF-8 CSVs and run the existing review tool.

~~~powershell
.venv/verify/Scripts/python.exe -m scripts.review_relevance_annotations --packet build/review/final/pilot/packet/private/packet.json --first evals/local/rater-a-completed.csv --second evals/local/rater-b-completed.csv --output-dir build/review/human-comparison
~~~

The tool rejects blanks, edited context and missing/duplicate tasks. Blank is
incomplete, never zero relevance. Unjudgeable images/context must be resolved by
the coordinator before final submission; do not silently remove rows. Any
required exclusions/data repair need a regenerated, consistent packet.

A human adjudicator resolves disagreements with reasons. Only then can the
existing exporter, preflight and ablation tools evaluate ranking agreement.
Because this pilot is all dev and scenario-based, its eventual scores cannot
be presented as held-out generalization or real-user/F1/F2 accuracy.
