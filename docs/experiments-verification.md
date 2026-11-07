# F3, paired comparison, and CI verification

Verified locally on 2026-11-07 using Python 3.13.7 on Windows.

## Implemented

- Four F3 variants: filtered random, vector-only, production hybrid, and production
  hybrid with diversity. All use the same frozen, eligible candidate pool.
- Shared catalog deduplication/aliases and storage-independent ranking types.
  Production score formulas, weights, ordering, and selection behavior are retained.
- Versioned analysis artifacts with the complete selected case set recorded.
- Strict paired-run input checks, per-field improvements/regressions, retained
  failures, configuration/model-version differences, latency and usage diagnostics.
- Lightweight test requirements, backend GitHub Actions, and offline smoke command.
- English intuition/schema/pipeline/code walkthrough in [experiments.md](experiments.md).

## Checks completed

| Check | Observed result |
| --- | --- |
| Full pytest suite with remote TCP connects blocked | 157 passed |
| Existing tests from the previous batch | All 122 retained |
| New experiment/CI regression cases | 35 |
| Ruff check | Passed |
| Ruff formatter check | 89 Python files already formatted |
| pip check | No broken requirements |
| Workflow syntax/expression validation | actionlint v1.7.7 passed |
| Offline CI smoke command | Two 10-case runs, 10 paired cases, 2 retrieval cases per variant |
| Direct paired-comparison CLI | Passed |
| Rescoring exported F3 predictions | Same metrics as the ablation report |
| Prompt files, query templates, vectors, seed catalog | Unchanged from the starting commit |
| Frontend | No changes |
| Root user SQLite files created | None |

actionlint was downloaded from its official GitHub release and checked against
the published SHA-256 checksum. Optional shellcheck/pyflakes integrations were
disabled because they were not installed on this Windows machine.

The full suite has one third-party DeprecationWarning from Starlette/AnyIO about
BlockingPortal. It does not fail checks; runtime dependencies were not upgraded
to address an unrelated warning.

## Evidence behind the checks

- Full hybrid/diversity results match production recommendation IDs, order,
  retrieval scores and final scores in temporary SQLite for fixed vectors.
- Filters, male aliases, duplicate images, stable vector ties, seeded randomness,
  unchanged global RNG state, and case-order independence are covered.
- Malformed vectors, synthetic/real metadata mismatch, stale templates, missing
  judgments, duplicate IDs, and subject split leakage are rejected.
- Exported judgments preserve the eligible pool, including the NDCG denominator.
- Paired tests cover recovered errors, regressions, unchanged mistakes, reordered
  evidence, missing/extra/duplicate cases, changed labels/inputs, and demo rejection.
- Unknown usage stays null; token deltas are restricted to complete pairs.
- Same requested model alias with changed resolved provider version is disclosed.
- Cached aggregate reports are ignored; metrics are recomputed from evidence.
- A fresh retrieval subprocess neither imports database modules nor reads DB
  configuration, and works without an API key.
- The smoke phase blocks a remote connect and restores its prior socket policy
  when a check fails.

## Artifacts

Ignored local artifacts:

- build/experiments/verified-batch-two/smoke-summary.json
- build/experiments/verified-batch-two/comparison-cli/comparison.md
- build/experiments/verified-batch-two/retrieval/report.md
- build/experiments/verified-batch-two/retrieval/rescored.json

Synthetic percentages and random-baseline wins are fixture outcomes. They do not
justify a claim that a ranker improves real fashion recommendations.

## Pending external validation

The workflow defines Ubuntu/Python 3.12 and 3.13 jobs. Only local Windows/Python
3.13 execution and workflow syntax were verified here; hosted jobs have not run.
No code was pushed, deployed, or published. Real embedding exports, independent
human relevance labels, and live Gemini measurements remain separate work.
