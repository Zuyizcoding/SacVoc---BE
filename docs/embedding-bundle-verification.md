# Corrected embedding bundle: measured local results

Date: 2026-11-08, Asia/Bangkok. The user authorized fixing projection and producing
a new synchronized bundle in separate output. The application was not switched.

## Actual output

Bundle: build/real-catalog/projected-v2/.

| Measurement | Result |
| --- | --- |
| Identity | fashion_clip / fashion-clip / 2026-11-projected-v2 |
| Saved query vectors | 240 finite normalized 512D vectors |
| CSV rows retained | 80 |
| Unique URLs audited | 62 |
| URLs downloaded, decoded and encoded | 59 |
| Encoded catalog rows | 77 |
| Unencoded rows | 3, retained with NULL vectors |
| Device-history rows | 0 |
| Application switched | false |

Rows 51, 55 and 62 still return HTTP 403 from static.zara.net. There was no
URL replacement or bypass. The 59 available URLs had 59 distinct downloaded
file hashes. Repeated URL references are preserved in the 80-row catalog.

| Occasion | Total rows | Encoded rows |
| --- | --- | --- |
| daily | 32 | 31 |
| work | 16 | 15 |
| party | 17 | 17 |
| date | 15 | 14 |

SQLite float32 image-vector norms range from 0.99999994 to 1.0000001.
All stored encoded rows share the new identity.

## Numerical evidence

The pinned FashionCLIP revision remains
7e3ba62ce16b379a1ab479346b66f192e76f51b7, with safetensors SHA-256
4977e3a54929eccf065ce449aeaf296f0e5cb6b28e8798c3c97d67cb2f6dafc9.
No weights were trained or changed.

Generation used CPU Torch 2.7.1+cpu and Transformers 4.52.4, eager attention,
the existing processor and 77-token text padding/truncation.

- All 240 saved queries matched text_model -> text_projection exactly once.
- Maximum query coordinate error: 2.9802322387695312e-08.
- Minimum cosine: 0.9999999999999996.
- All 59 encoded URLs matched vision_model -> visual_projection exactly once.
- Maximum normalized image coordinate error before storage: 0.0.
- The gate remains cosine >= 0.99999 and coordinate error <= 0.0001.

The new encoder still fails to reproduce the legacy query bundle: 0/240 pass.
This is expected evidence of the corrected geometry, not a reason to loosen the
gate. The old diagnostic report reproducing a second projection is preserved.

## Cross-SDK verification

The installed global environment also provided Transformers 5.3.0 and Torch
2.10.0. The same local checkpoint was loaded without downloads, using the repaired
loader and feature-output wrappers. All 240 queries and all 59 images matched
the saved v2 bundle within the unchanged tolerance:

- Maximum query coordinate error: 4.172325134277344e-07.
- Maximum image coordinate error: 4.023313522338867e-07.

Evidence:
build/real-catalog/projected-v2-verification/sdk5-model-check.json.

Transformers 5 reported two unexpected legacy position_ids checkpoint entries.
The complete numerical checks above passed despite those loader differences.
This does not assert support for every future SDK version.

## Saved-data and serving verification

build/real-catalog/projected-v2-verification/report.json records:

- All manifest artifact hashes matched; baseline inputs remained unchanged.
- SQLite integrity passed; no device history was present.
- All 240 season/body/occasion profiles had eligible encoded male outfits.
- Runtime ORM and snapshot-export filtering selected identical candidate IDs.
- Candidate pools contained 11 to 28 unique image URLs after filtering.
- Each profile returned six results, with no repeated image URL.
- Repeated ranking returned identical IDs and scores.
- Cosine values were finite and final match scores remained in [0, 1].
- The read-only verification left catalog.db byte-for-byte unchanged.

These checks establish software/data consistency, not recommendation quality.
Independent relevance labels are still required for a fashion-quality benchmark.

## Regression and static checks

- Full lightweight backend suite: 259 passed, 2 optional modules skipped.
- Optional CPU environment: 24 passed across learning retrieval, real tiny-CLIP
  feature-contract checks and compatibility regression tests.
- The real tiny-CLIP feature-contract check also passed on Transformers 5.3.0.
- Ruff lint and formatting passed for 129 Python files.
- Dependency compatibility check passed in the optional environment.
- Existing Starlette/AnyIO and optional global SWIG deprecation warnings remain.

Regression tests cover both SDK containers, no second projection, idempotent
wrapping, bound loader keyword handling, runtime fallback behavior, v1/v2 mixing
rejection, independent query/image reference failures, rollback, partial source
availability, refused overwrite, manifest tampering and all-profile serving.
Ordinary tests use fake encoders/HTTP and temporary DBs; tiny CLIP tests initialize
random small local weights and download nothing.

## Preserved baseline

The before/after hashes recorded in
build/real-catalog/baseline-before-v2.json matched for all seven protected files:

- outfitseed.csv.
- data/query_vectors.json and data/query_vectors.meta.json.
- data/outfit_vectors.meta.json.
- app/services/recommendation/query_builder.py.
- build/real-catalog/run-01/catalog.db and its summary.json.

The existing application DB/query configuration, ranking formula, weights,
HTTP endpoints, device identity, F1/F2 prompts and Flutter UI were not changed.
No Gemini request, training, application deployment or automatic data migration
was performed. The old source audit and compatibility diagnostics remain intact.
