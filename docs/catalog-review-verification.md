# F3 review preparation: local verification

Date: 2026-11-08, Asia/Bangkok. All outputs are under build/review/final/.
The source is build/real-catalog/projected-v2/, identity
fashion_clip / fashion-clip / 2026-11-projected-v2.

## 1. Actual FastAPI route checks

api/report.json records 240 supported profiles, each requested twice through the
unchanged FastAPI application using TestClient. All 80 outfit-detail rows were
checked, including static records with unavailable source images.

Thirteen additional scenario checks passed:

- Missing both F1/F2, only F1, and only F2: HTTP 400 with existing error/missing fields.
- Missing device header, short device ID, invalid occasion, unsupported gender:
  HTTP 422.
- F3 readiness: HTTP 200 and recommendation.ready=true.
- Missing detail: HTTP 404.
- Omitted occasion: the query defaults to daily.
- Mixed versions inside the catalog, a uniform query/catalog version mismatch,
  and an empty catalog: existing HTTP 200 failure envelope with
  RECOMMENDATION_NOT_READY.

Successful results use precomputed_query_vector, contain six distinct image URLs,
match profile/occasion eligibility, and repeat identical response JSON.
These are software fixtures in a disposable DB, not AI-generated analyses.
No Gemini or runtime encoder call occurred; external TCP was blocked.

The worker ran in a separate process with explicit settings. Its SQLite copy was
removed after completion. Source DB/query/manifest hashes remained unchanged.

## 2. Catalog audit and gallery

catalog/report.json and catalog/gallery.html cover all 80 CSV rows.

| Finding | Count |
| --- | --- |
| Structural errors | 0 |
| Review warnings | 12 |
| Shared-image informational groups | 2 |
| Available local image files | 59 |
| Image elements in the 80-card gallery | 77 |
| Source images unavailable | 3 |

Warnings break down into seven missing body-shape metadata flags, three
unavailable images and two shared-image product metadata conflicts.

The shared-image groups contain these CSV data rows:

- 9, 15, 16, 27, 29, 30, 32, 34, 35, 41, 56.
- 11, 38, 46, 49, 53, 54, 64, 71, 79.

Those 20 rows share only two image URLs while differing in name, category,
product URL and price. A coordinator must determine which mappings are valid;
the audit made no changes and does not assert visual truth.

Missing body_shape_bias occurs on rows 45, 51, 55, 62, 66, 73 and 77. All seven
are footwear, so empty shape metadata may be intentional. The existing fallback
was preserved rather than filling arbitrary labels.

Rows 51, 55 and 62 remain unencoded after the previous HTTP 403 source outcomes.
This stage used frozen local images and made no new source-image requests.

## 3. Independent-rating pilot

pilot/summary.json records:

- 20 explicitly hypothetical profile scenarios, all dev.
- All 12 color seasons represented (1 or 2 cases each).
- All 5 body shapes represented four times.
- All 4 occasions represented five times.
- 325 eligible tasks per reviewer, 650 blank ratings in total.
- Same task sets, independently shuffled orders, blank grades and notes.
- Two separate reviewer ZIPs with original local image bytes.
- Packet bound to the exact catalog fingerprint and source bundle manifest.
- Zero completed relevance ratings.

Both reviewer pages contain 325 cards and local image references, with no scores,
confidence or bias metadata. Each folder has all 59 local images and can be used
after extracting its ZIP without a network connection.

The materials remain draft until the coordinator reviews the catalog conflicts
and agrees on the rubric. They do not constitute an independently judged
benchmark or measured recommendation accuracy.

## Tests and static checks

- 11 new offline regressions passed, including a complete API subprocess run.
- Full lightweight backend suite: 270 passed, 2 optional modules skipped.
- Separate CPU environment: 24 passed across retrieval/feature/compatibility checks.
- Ruff lint and formatter passed for 138 Python files.
- Existing Starlette/AnyIO deprecation warning remains.

New regressions cover source preservation, missing-analysis/error envelopes,
version rejection, local image hashes and path confinement, exact byte copies,
HTML escaping/no scripts, profile coverage, blank forms, unchanged reviewer
context, ZIP isolation and mismatched/structurally invalid audit rejection.

The actual generated HTML pages were parsed and all local image paths/checksums
were validated; evidence is html-check.json. Browser visual QA was NOT completed:
the Computer Use runtime stopped because it could not confidently determine
Chrome's current URL for policy enforcement. No further browser interactions
were attempted. Static HTML/asset checks do not prove browser layout.

## Handoff and remaining work

Baseline checksums still match for the seven protected files recorded before v2,
including the original isolated DB. The v2 manifest also still matches. No source
data, production settings, ranker weights, F1/F2 prompts or Flutter UI changed.

The preparation needed before human review is complete. Remaining inputs are:

1. Coordinator decisions about shared-image mappings, footwear metadata and rubric.
2. Two independent reviewers and explicit human adjudication of disagreements.
3. Valid replacement sources for the three blocked images if 80-row coverage is required.
4. Consented photos, reference labels and securely configured Gemini for real F1/F2 evaluation.

Current workspace presence checks found no .env, no GEMINI_API_KEY in the parent
process environment, and no files under evals/local/. Flutter was not found on
PATH; this does not prove that no SDK exists elsewhere on the machine.

The agent can validate completed files, list disagreements, export the frozen
snapshot and run ablations after those inputs exist. Further engineering such
as an opt-in local demo or hosted CI is optional and does not substitute for
human relevance judgments. The application has not switched to v2 automatically.
