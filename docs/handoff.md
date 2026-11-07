# Handoff: what is ready, what the agent can still do, and what needs people

Audit date: 2026-11-07. Evidence is local to this workspace, not an assertion about
datasets or credentials you may have elsewhere. The current tools are ready for
real-data work: schema checks, inference traces, paired comparisons, F3 ablation,
preflight, read-only snapshots, independent annotation, and container runtime.

## 1. Observed input readiness

The local audit at build/handoff/readiness.json found:

| Input | Observed state |
| --- | --- |
| Gemini key / .env | Neither configured/present |
| Real evaluation files in evals/local/ | None |
| Referenced sample F1/F2 image paths | 0 of 4 exist |
| Root SQLite catalog files | None |
| Saved query vectors | 240 templates, signature matches |
| Hosted workflow result for these changes | Not confirmed |

Only presence was inspected; no credential value is recorded. A present key would
still need model/quota validation. A file count does not prove consent or
annotation independence. The sample benchmark JSON files are examples, not a
completed real evaluation dataset.

## 2. Corrected separate embedding bundle is ready for review

Follow-up date: 2026-11-08. After explicit authorization, the projection adapter
was repaired and a new versioned bundle was generated in
build/real-catalog/projected-v2/. All 240 queries and 59 available image URLs
passed independent one-projection reference checks; 77/80 catalog rows have
image vectors. Zara rows 51, 55 and 62 remain unencoded after HTTP 403 responses.

The saved bundle passed read-only serving/filter/determinism checks for all
240 profiles. It also reproduced on the installed Transformers 5.3.0 SDK after
generation with 4.52.4. No Gemini calls or model training were needed.

Committed embeddings/CSV/templates and the original isolated DB remain
unchanged. The application has not switched to the new bundle. See
[embedding-bundle.md](embedding-bundle.md) and
[embedding-bundle-verification.md](embedding-bundle-verification.md) for evidence
and commands. The old blocked run remains historical evidence.

The next preparation batch is now complete under build/review/final/: real
FastAPI integration checked all 240 profiles and 80 details in a disposable DB;
a coordinator gallery covers all 80 rows; and a 20-scenario, all-dev pilot has
325 blank tasks per reviewer with separate offline ZIPs.

The catalog audit found two shared-image metadata-conflict groups spanning 20
rows. Seven footwear rows have no body_shape_bias, which may be intentional.
These need coordinator review before annotation begins. The pilot uses
hypothetical scenarios, not real subjects or model-predicted F1/F2 labels.
See [catalog-review-pilot.md](catalog-review-pilot.md) and
[catalog-review-verification.md](catalog-review-verification.md).

Human ratings and valid replacement sources are still required before claiming
measured quality or full image coverage. The agent can process completed files
and run experiments afterwards. Optional local-demo or hosted-CI engineering
remains possible; it does not replace independent judgments.

## 3. Inputs that require the project owner or human reviewers

1. Configure a usable Gemini key securely on the machine and choose the live-run
   model/quota/spending scope. Do not paste the key into chat or commit .env.
2. Supply consented F1/F2 photos, pseudonymous subject IDs, dev/test assignments,
   and independently prepared reference labels. The agent can validate and format
   them, but cannot establish visual truth or permission on your behalf.
3. Arrange two reviewers to complete F3 relevance ratings independently. A
   designated adjudicator must resolve differences with a written rationale.
   Software can compare files, not replace independent human judgments.
4. Identify your personal contribution and any application-specific claims before
   publishing CV/portfolio material. Synthetic test rates are not AI accuracy.

For F1, four primary seasons can be the initial target; optional sub-season
labels must use the declared taxonomy. For F2, use the existing male-shape labels.
The annotation protocol should describe uncertainty and disagreement rather than
calling subjective judgments objective ground truth.

## 4. What the agent can run after those inputs exist

- Preflight images/splits/support and produce issues for review.
- Prepare blinded F3 packets from a compatible catalog without ranking leakage.
- Validate reviewer CSVs, report differences, and export the adjudicated request.
- Export/freeze vectors and run filtered-random/vector/hybrid/diversity comparisons.
- Run live F1/F2 inference, retaining failures, repairs, usage and latency.
- Compare frozen baseline/candidate runs and write an error-analysis report.
- Produce accurate English project/demo/CV descriptions based on observed results.

See [annotation-and-runtime.md](annotation-and-runtime.md) and
[evaluation.md](evaluation.md) for the exact commands. Real Gemini execution
sends the selected images to the provider and may incur charges; current tests
used no real Gemini request.

## 5. Decisions for the measured phase

GEMINI_TIMEOUT remains per provider attempt. F1/F2 repair/best-effort can increase
end-to-end duration beyond Flutter's current 75-second receive timeout. Retry,
fallback and UI policies were deliberately preserved. Use live timing evidence
to choose a coordinated end-to-end budget before claiming a production latency
target; changing that policy is separate from these refactors.

Model-generated confidence is not calibrated correctness probability. Unit tests,
synthetic grades and a passing Docker health check do not establish AI quality.
Real examples and independent judgments are the next source of useful evidence.
