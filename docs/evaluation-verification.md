# SDK and evaluation implementation verification

Verified locally on 2026-11-06, using Python 3.13.7 (project requirement: 3.12+).

## Scope

- Replace the legacy Gemini SDK with google-genai==1.56.0.
- Preserve the analyze() interface, retry budgets/backoff, breaker lifecycle, and
  existing API persistence/fallback contracts.
- Add explicit live/demo/evaluation execution policies.
- Add a database-free F1/F2 runner with per-call/per-attempt evidence.
- Add classification and independent F3 relevance metrics.
- Explain intuition, schemas, pipeline, commands, and limitations in English in
  module docstrings and [the evaluation guide](evaluation.md).

The previous 62 backend tests remain, with SDK-boundary imports/fakes migrated.
There are 60 additional cases, including additions to the API contract tests.

## Checks completed

| Check | Result |
| --- | --- |
| pytest -q | 122 passed |
| ruff check app scripts tests | Passed |
| ruff format --check app scripts tests | 76 files already formatted |
| pip check in a fresh isolated environment | No broken requirements |
| Legacy SDK/API-core packages in that environment | Neither installed |
| F1/F2 synthetic CLI | Completed all 10 scripted cases; JSONL/CSV/JSON/Markdown artifacts generated |
| F3 relevance-export CLI | Completed; missing predictions retained as misses |
| Git diff whitespace check | Passed |
| Prompts, recommendation implementation, vectors, and seed catalog | No changes relative to the starting commit |
| Frontend working tree | Unchanged |

The clean environment was created under the ignored .venv/ directory and installed
production requirements plus pytest and Ruff. It did not depend on globally
installed Google packages.

One third-party DeprecationWarning remains: Starlette's test client references
anyio.abc.BlockingPortal. It does not fail tests. Dependencies other than the
requested Gemini SDK were not upgraded in project requirements.

## Regression coverage

- Real pinned SDK serialization over fake HTTP: image MIME/base64, system/user
  prompts, nested nullable schemas, confidence bounds, thinking and safety config.
- SDK timeout unit conversion and exactly one SDK attempt per application attempt.
- Quota/busy/server/transport failures, retry order/backoff, fail-fast, cooldown,
  success reset, and invalid output not opening the breaker.
- Empty/malformed/non-object/truncated JSON, schema errors, safety blocks and
  unspecified safety feedback.
- Initial inference, missing-field repair, clothing best effort, all four F2
  logical-call stages, incomplete results, and unchanged API error contracts.
- Live fallback stored in temporary history; evaluation failure never converted
  to a demo; explicit demo never calling AI.
- Concurrent trace isolation and restoration after exceptions.
- A fresh subprocess running without a key, importing neither database module
  nor persistence helpers, with an existing temporary SQLite sentinel unchanged.
- Subject split leakage, duplicate IDs, inconsistent taxonomy labels, human
  consent/script validation, output overwrite protection, and read/photo errors.
- Hand-computed classification/ranking metrics, missing predictions, zero-support
  classes, undefined NDCG, unjudged IDs, duplicates, and percentile interpolation.

## Local artifacts

The verified runs are available in ignored build output:

- build/evaluation/final-analysis/
- build/evaluation/final-ranking/report.json

Failures in the synthetic example are intentional. Their percentages are fixture
properties, not model accuracy or production performance.

## Limits

No real Gemini requests, human annotation, model downloads, catalog changes, or
user DB writes were performed. Live model availability, visual quality, latency,
and cost still need credentials and independently labeled, consented data.
Direct Pydantic schemas and thinking-budget support differ from the legacy
adapter's silent configuration fallbacks; live generations have not been claimed
to be identical. The runner retains per-attempt timeout semantics, not a new
whole-analysis deadline. No frontend code changed, so Flutter checks were outside
this implementation's scope.
