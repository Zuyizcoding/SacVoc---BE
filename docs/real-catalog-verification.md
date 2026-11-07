# Real catalog preparation: local verification

Date: 2026-11-07 (Asia/Bangkok). This is an actual public-image/model check,
not a synthetic catalog experiment. No Gemini call or user DB access occurred.

## Observed result

**Image encoding was correctly blocked.** The normal FashionCLIP text encoder
does not reproduce the committed query vectors. The separate DB contains all
80 rows and zero image vectors; it must not be deployed as a ready F3 catalog.

Artifacts are ignored local files under build/real-catalog/run-01/:

| Artifact | Result |
| --- | --- |
| catalog.db | 80 outfits, 0 encoded, 0 analysis-history rows |
| sources.json | Initial audit of all 62 unique image URLs |
| source-recheck.json | One normal retry of the single timed-out URL |
| encoder.json | Pinned checkpoint, hashes, packages and processor details |
| compatibility.json | All 240 normal text vectors failed the numerical gate |
| projection-diagnostic.json | Extra-projection hypothesis reproduced all 240 |
| summary.json | Initial run status and unchanged input-file hashes |

SQLite PRAGMA integrity_check returned ok. The initial source audit found
58 available URLs representing 76 CSV rows. The one timed-out URL subsequently
loaded and decoded successfully, raising observed coverage to **59/62 URLs and
77/80 rows**. Original sources.json and summary.json remain unchanged so their
initial counts do not silently become retry results.

Three URLs from static.zara.net returned HTTP 403. Their 1-based CSV data-row
IDs are **51, 55 and 62**. No alternate URLs, access-control workarounds, or
catalog edits were used. The successfully retried source is row 2.

The initial 58 downloaded files had 58 distinct file hashes and 58 distinct
decoded-RGB hashes. There are 18 repeated URL references in the original
80-row CSV; these are retained rather than silently deleted. Availability and
exact hashes do not prove correct product descriptions or image licensing.

## Encoder identity and numerical gate

- Repository: patrickjohncyh/fashion-clip.
- Revision: 7e3ba62ce16b379a1ab479346b66f192e76f51b7.
- model.safetensors: 605157890 bytes.
- SHA-256: 4977e3a54929eccf065ce449aeaf296f0e5cb6b28e8798c3c97d67cb2f6dafc9.
- Weight hash matches the public Hugging Face LFS metadata.
- CPU Torch 2.7.1+cpu, Transformers 4.52.4, NumPy 2.2.2, Pillow 11.1.0.
- Safetensors only, eager attention, eval/inference mode.
- Original 77-token max-length padding and truncation.

For the normal projected text features:

| Measurement | Observed |
| --- | --- |
| Queries checked | 240 / 240 |
| Queries passing both thresholds | 0 / 240 |
| Minimum cosine | -0.1407180115 |
| Mean cosine | -0.0541409140 |
| Maximum coordinate error | 0.5604058504 |

The required minimum cosine is 0.99999 and maximum coordinate error is 0.0001.
This mismatch is far beyond floating-point rounding.

## Diagnosis: the saved texts reproduce a second projection

Applying the same model's text_projection to its already projected text
features reproduces all 240 committed query vectors:

- 240/240 pass the diagnostic comparison.
- Maximum coordinate error: 1.7136335372924805e-07.
- Minimum reported cosine: 0.9999999072448847.
- Small reported cosines above 1 can occur because the diagnostic compares
  float32-normalized vectors; they reflect rounding, not greater similarity.

The existing app/utils/fashion_clip_compat.py helper applies a projection when
pooler_output has the projection's input dimension. For this model both hidden
text and projected text dimensions are 512, so that dimension check cannot
distinguish them. Transformers 5.3.0 already places the projected result in
pooler_output. The current runtime fallback in embedding_store.py contains a
similar check and belongs in a coordinated repair.

This is strong numerical evidence of the historical generation path. There is
no historical package lock attached to the old vectors, so the exact original
runtime version is not asserted. The diagnostic is NOT an approved replacement
encoder: reproducing an incorrect extra text projection would not repair
alignment with the official image encoder.

## Preserved inputs

| Input | SHA-256 |
| --- | --- |
| outfitseed.csv | ac6d4a5b03fc8a9064129e38ed297231f561a67d3c55faad43b48e9016095ede |
| data/query_vectors.json | 07c0464dd68264b188ebede50e7e58165b1269c74ab3d76c3b67159d9b7347e2 |
| data/query_vectors.meta.json | da077db8f7c70213c9eb9f456fe4958f4fc97424063365f1088d2077da30434b |

The query signature remains
9c04976c9db0fc3619050c0da9820c187d38c7d6314606f217ca94e9db191fb4.
Committed outfit metadata, CSV, query data and query templates were not replaced.

## Software checks

- 15 new offline regression checks passed: every-query matching, malformed or
  stale query bundles, zero/nonfinite/wrong-dimension vectors, source deduplication,
  image decoding, HTTP denial/timeout, the compatibility gate, separate DB row
  preservation, refusal to overwrite output, fatal cache-write errors, and rollback on late encode failure.
- Full lightweight backend suite: 246 passed, 1 optional Torch module skipped.
- Ruff lint passed; all 122 Python files passed the formatter check.
- One existing Starlette/AnyIO deprecation warning remains.

Tests use fake HTTP, fake encoders and temporary DBs. They download no model,
make no Gemini request and do not treat synthetic matches as real AI accuracy.
Real image/model measurements above were run separately in the optional local
environment.

## Required next decision

Do not bypass the gate or relabel vectors as compatible. The recommended next
scope is to fix projected-feature handling with regression coverage, generate a
new versioned query/image bundle in separate output, and validate it together.
This changes the saved query embeddings and requires changing the earlier
preservation constraint. Keep the current artifacts as a comparison baseline.

After that decision, the agent can encode the 77 currently available rows and
prepare the independent relevance workflow. The three denied image sources
need valid replacement sources before full 80-row coverage can be claimed.
