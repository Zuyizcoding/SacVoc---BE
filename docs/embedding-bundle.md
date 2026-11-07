# Corrected, separate query/image embedding bundle

This follow-up was authorized after diagnosing the legacy double projection.
The original preservation constraint now allows a new bundle in separate output.
Committed query data, template text/signature, CSV, old DB and application paths
remain unchanged. The corrected local bundle is not automatically activated.

Read [the measured results](embedding-bundle-verification.md) alongside this guide.
The previous [blocked run](real-catalog-verification.md) remains historical evidence.

## Intuition

F3 compares text and image coordinates. An extra projection on text changes
those coordinates; matching dimensions or metadata cannot repair the geometry.

CLIPModel's feature methods already apply their learned projections. Transformers
4 returns tensors, whereas the checked Transformers 5 API puts the projected
tensor in pooler_output. The compatibility layer must unwrap it, not project it
again. Both hidden and projected text can be 512-dimensional, so a size-based
decision is invalid.

The new builder verifies the feature API against another path:
text_model -> text_projection exactly once, and
vision_model -> visual_projection exactly once.
This avoids a test that merely repeats the same potentially faulty adapter.

## Schema and files

The bundle identity is:

~~~json
{"backend":"fashion_clip","model":"fashion-clip","version":"2026-11-projected-v2"}
~~~

Version v2 distinguishes corrected geometry from the legacy 2026-05-mens-v1
vectors. The template signature is unchanged: the 240 query texts are identical,
but their numerical embeddings differ.

| Artifact | Purpose |
| --- | --- |
| query_vectors.json | 240 normalized, finite 512D text vectors |
| query_vectors.meta.json | Identity, count, template signature and timestamp |
| catalog.db | Existing ORM schema, all CSV rows, encoded images where available |
| sources.json and images/ | Source audit and original downloaded bytes |
| encoder.json | Pinned checkpoint, processor hashes and package versions |
| query-reference-check.json | Every saved query versus one explicit projection |
| image-reference-check.json | Every encoded URL versus one explicit projection |
| baseline-comparison.json | Diagnostic difference from the unchanged legacy queries |
| readback-check.json | Validate saved JSON/DB rather than only in-memory arrays |
| manifest.json | Artifact checksums and preserved input checksums |
| summary.json | Outcome, coverage and explicit application_switched=false |

The final manifest is published only after successful validation. A failed run
retains diagnostic evidence with status=failed and cannot masquerade as a ready
bundle. Existing output directories are refused.

All source rows remain in the DB. Unavailable images have NULL vectors and no
embedding metadata. Repeated URLs are encoded once, and their rows share the same
vector. The normal F3 filtering and duplicate-image rules remain unchanged.

## Pipeline and code map

1. Read and validate the baseline. Record hashes before writing new output.
2. Generate all 240 queries using the pinned local FashionCLIP checkpoint.
3. Read the new JSON back and verify every vector against one explicit text
   projection. The numerical thresholds were not loosened.
4. Compare the encoder with the old queries for diagnosis only. A mismatch is
   expected; reproducing the historical bug is not a new acceptance target.
5. GET/decode/audit each public image URL using normal TLS verification.
6. Seed a fresh DB using the existing metadata inference rules.
7. Verify each image feature against one explicit visual projection before
   storing vectors. Commit them in one transaction; roll back on a late failure.
8. Validate stored vectors and matching identities through the existing read-only
   exporters, then publish checksums.
9. Separately verify all 240 supported serving profiles without loading a model
   or creating device history.

Implementation:

- [fashion_clip_compat.py](../app/utils/fashion_clip_compat.py): SDK output
  unwrapping and legacy loader keyword compatibility.
- [encoder.py](../app/catalog_preparation/encoder.py): pinned local weights,
  normal feature methods and independent reference paths.
- [bundle.py](../app/catalog_preparation/bundle.py): explicit build sequence.
- [workflow.py](../app/catalog_preparation/workflow.py): existing isolated seeding
  and transactional image writes, with an explicit embedding identity.
- [validation.py](../app/catalog_preparation/validation.py): runtime/export
  filtering, metadata, score bounds and deterministic ranking checks.

The legacy preparation CLI still checks the old bundle and still blocks on
incompatibility. Its gate was not bypassed. Corrected runtime and legacy encoder
scripts now identify newly generated vectors as v2, so their outputs cannot
silently mix with a v1 catalog.

## Run and verify

Use the optional environment from [real-catalog.md](real-catalog.md), with
requirements-catalog.txt and CPU Torch installed. No server dependency upgrade
or model training is required.

~~~powershell
.venv/catalog/Scripts/python.exe -m scripts.build_embedding_bundle --output build/real-catalog/projected-v2 --model-directory build/catalog-model
.venv/verify/Scripts/python.exe -m scripts.verify_embedding_bundle --bundle build/real-catalog/projected-v2 --output-dir build/real-catalog/projected-v2-verification
~~~

Choose fresh output paths for another run. The build downloads public images;
the verification command uses no network, model or Gemini call. It validates
manifest hashes before opening the DB read-only.

The verification report records one result for each of the 240 profiles:
eligible candidate count, selected IDs, match scores and cosine scores. It
compares the actual ORM candidate loader with the snapshot exporter, reruns the
same ranker for determinism, and checks duplicate images and result counts.
It does not invent reference labels or calculate recommendation accuracy.

## Review before activation

The application still uses its original default DB/query paths. Future activation
needs both DATABASE_URL and F3_QUERY_VECTORS_PATH to point to the same new bundle.
Restart the process so cached queries/manifests cannot retain the old bundle.
Changing only one path must fail version consistency checks.

Before such activation, review the missing image sources and collect independent
relevance ratings with the existing annotation tools. A correctly aligned vector
space is necessary, but does not establish fashion suitability. The ranker
formula and weights were preserved; input vectors changed, so scores and ordering
can change. Keep the old bundle as a reproducible legacy baseline.
