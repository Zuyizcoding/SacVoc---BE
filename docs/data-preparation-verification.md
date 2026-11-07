# Data preparation and PyTorch verification

Verified locally on 2026-11-07 with Python 3.13.7 on Windows.

## Delivered

- Analysis/retrieval dataset preflight with JSON and Markdown reports.
- Exact encoded-byte and oriented decoded-RGB duplicate detection.
- Case and distinct-subject label support; explicit cross-split blocking.
- Read-only SQLite catalog export with complete saved-query validation.
- CPU PyTorch retrieval helpers, an explained notebook and an execution script.
- Separate optional learning requirements; backend/CI still have no Torch.
- A synthetic preparation smoke command integrated into the backend workflow.

Start with the English [guide](data-preparation.md).

## Observed checks

| Check | Result |
| --- | --- |
| Lightweight backend suite, external TCP connects blocked | 190 passed, 1 skipped |
| Previous backend tests | All 157 retained |
| New backend preparation cases | 33 passed |
| Optional learning tests, separate environment | 13 passed |
| Notebook execution | All 7 code cells completed |
| Ruff lint | Passed |
| Ruff formatter | 102 Python files already formatted |
| pip check, backend and learning environments | No broken requirements |
| Workflow syntax/expression check | actionlint passed |
| Preparation smoke | Passed: generated image, preflight, export, ablation |
| Committed prompts/templates/query vectors/catalog CSV | Unchanged |
| Frontend | No changes |
| Actual user DB / Gemini / model encoder usage | None |

The skip is the optional Torch test module in the lightweight backend environment,
not a failed backend feature. The learning environment has torch==2.7.1+cpu and
runs its 13 tests explicitly. The source notebook has no saved output; its
executed copy and runtime-version summary are separate artifacts.

Two third-party warnings were observed: the existing Starlette/AnyIO deprecation
in backend tests, and a Windows Proactor/ZMQ warning while executing Jupyter.
ZMQ used its selector-thread fallback and all notebook cells completed.

## Regression evidence

- Exact file duplication and equal pixels with different PNG metadata are found
  across dev/test despite different subject IDs; source images stay unchanged.
- EXIF orientation is included in fingerprint layout without writing an image.
- Missing, malformed, truncated and undersized files produce explicit issues.
- Invalid manifests skip inspection rather than silently dropping bad records.
- Case counts and distinct subjects are verified separately.
- Retrieval preflight uses the same eligible judgment pool as ablation.
- SQLite mode=ro rejects writes and never creates a missing database.
- Export does not migrate a missing schema or include private analysis history.
- Source DB bytes and query/request files are unchanged after a successful export.
- Committed WAL updates are visible without changing the main DB file.
- Mixed/dummy metadata, malformed/non-finite/non-unit vectors, stale signatures,
  incorrect query count, missing templates and unknown judgments fail explicitly.
- Unencoded rows are counted and excluded, never converted to synthetic candidates.
- A subprocess works with invalid application DATABASE_URL and no Gemini key,
  and imports neither the application DB module nor the runtime encoder store.
- PyTorch scores match hand calculations and NumPy; stable ties, batching,
  invalid tensors and no-gradient behavior are covered.
- The notebook also checks score/ID parity with production vector-only retrieval.

## Local artifacts

Ignored output:

- build/preparation/verified-smoke/smoke-summary.json
- build/preparation/verified-smoke/export/snapshot.json
- build/preparation/verified-smoke/export/provenance.json
- build/preparation/verified-smoke/ablation/report.md
- build/learning/verified-notebook/executed.ipynb
- build/learning/verified-notebook/summary.json

All catalog/image/query fixtures were generated under build output or temporary
test directories and explicitly used as software checks. No committed embeddings
were regenerated or replaced.

## Limits

Exact hashes do not identify every perceptually similar photo. Readiness cannot
prove consent, independent annotation, sufficient sample size or fashion accuracy.
SQLite may use normal read-side locking/shared-memory sidecars; an offline copy
is appropriate when even source-directory sidecar activity is unacceptable.

Real snapshot export still requires real catalog embeddings and independent
relevance labels. This batch does not produce such labels or train a model.
The GitHub workflow was validated locally; hosted jobs were not run or published.
