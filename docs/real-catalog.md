# Preparing a real catalog in a separate DB

The original run (run-01) was **blocked before image encoding**, not ready for F3.
The authorized follow-up produced a [corrected separate bundle](embedding-bundle.md);
its [verification results](embedding-bundle-verification.md) supersede that blocker for the new data.
See [measured results](real-catalog-verification.md). Source availability and
embedding compatibility are separate gates; both need evidence.

## Intuition

CLIP places text and images in a shared coordinate system. Two 512-dimensional
unit vectors can come from different coordinate systems. Matching model labels
or valid norms cannot establish semantic compatibility.

This tool re-encodes all 240 existing query texts using a pinned official
FashionCLIP checkpoint. Only a numerical match allows image encoding. It never
changes the committed CSV, query vectors, metadata, prompts, or ranking rules.
A failure leaves every catalog vector NULL instead of filling it with dummy data.

## Schema and implementation map

| File/module | Responsibility |
| --- | --- |
| scripts/prepare_real_catalog.py | Parse paths and report the outcome |
| app/catalog_preparation/workflow.py | Present the business steps in order |
| app/catalog_preparation/sources.py | GET, size limit, image decode, hashes and cache |
| app/catalog_preparation/compatibility.py | Exact query coverage and numerical gate |
| app/catalog_preparation/encoder.py | Pinned CPU model, processor and provenance |
| scripts/diagnose_query_projection.py | Investigate a failed gate without encoding images |
| scripts/seed_outfits_csv.py | Existing metadata inference and DB schema rules |

Each output directory contains:

- catalog.db: the real ORM schema, all CSV rows in order, no device history.
- images/: downloaded original image bytes, addressed by their SHA-256.
- sources.json: one record per unique URL, CSV row IDs, HTTP/decode outcome,
  dimensions, hashes and local cache filename.
- encoder.json: exact model revision, weight/processor hashes, package versions
  and preprocessing choices.
- compatibility.json: pass/fail and numerical comparison for every query.
- summary.json: status, source counts, encoded/unencoded counts and input hashes.

Row IDs count data rows from 1, excluding the CSV header. The initial CSV must
have non-empty unique names and product URLs. This prevents the existing upsert
rules from collapsing separate rows. A fresh output directory is mandatory.

## Pipeline

1. Validate the CSV and the committed query bundle, including all template keys,
   signature, identity, finite dimensions and existing unit norms.
2. GET each distinct image URL with four workers and a 20-second socket timeout.
   Read at most 12 MiB, decode it, record dimensions and file/RGB hashes.
3. Seed a new DB using the existing CLI in a child process with an explicit
   DATABASE_URL. Check row order and absence of analysis history.
4. Load only the pinned safetensors checkpoint and its local processor.
5. Re-encode all 240 texts with the upstream 77-token padding/truncation.
6. Require every query to have cosine >= 0.99999 and maximum coordinate error
   <= 0.0001. These thresholds tolerate rounding, not a different geometry.
7. On success, encode each available URL once and reuse its vector for rows
   referencing that URL. Recheck cached byte hashes before inference.
8. Commit all image vectors in one transaction. Unavailable source rows stay
   present with NULL vectors. A later encoding failure rolls back vector writes.

HTTP 403 remains a source failure; the tool does not bypass access controls or
replace the URL. Technical decoding does not establish that the image depicts
the named product. Inferred fit/color/formality fields retain the existing seed
heuristics and still need review before making fashion-quality claims.

Unlike the evaluation photo preflight, RGB conversion here does not apply an
additional EXIF transpose: it preserves the original image encoder convention.

## Reproduce in an optional environment

Do not install Torch/model tooling into the lightweight server environment.

~~~powershell
python -m venv .venv/catalog
.venv/catalog/Scripts/python.exe -m pip install -r requirements-catalog.txt
.venv/catalog/Scripts/python.exe -m pip install -r requirements-learning-torch.txt
~~~

Download the public checkpoint into ignored build output:

~~~powershell
.venv/catalog/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('patrickjohncyh/fashion-clip', revision='7e3ba62ce16b379a1ab479346b66f192e76f51b7', local_dir='build/catalog-model', allow_patterns=['*.json','merges.txt','vocab.json','model.safetensors'], ignore_patterns=['onnx/*'])"
.venv/catalog/Scripts/python.exe -m scripts.prepare_real_catalog --output build/real-catalog/new-run --model-directory build/catalog-model
~~~

The local loader performs no network lookup and checks weight SHA-256 before
loading. No alternate encoder is tried on failure. A fresh run may need about
605 MB for weights plus processor files, the optional CPU environment and images.

Exit codes are 0 for an encoded catalog, 2 for an incompatible encoder, and 3
when compatibility passed but no available image could be encoded. A status of
encoded can still represent partial coverage; inspect unencoded_rows and sources.
Other execution errors raise normally and do not turn into successful reports.

TLS verification remains enabled. This Windows run encountered certificate-bundle
retry errors in huggingface_hub; downloading the same pinned public files through
Python urllib with the OS trust store succeeded. Do not use verify=False or
disable certificate checks.

## Investigate the current mismatch

Run the separate, explicitly diagnostic comparison:

~~~powershell
.venv/catalog/Scripts/python.exe -m scripts.diagnose_query_projection --model-directory build/catalog-model --output build/real-catalog/new-run/projection-diagnostic.json
~~~

It applies text_projection a second time to already projected text features,
then compares the result with the saved queries. Even a perfect diagnostic
match does not open the image-encoding gate. Applying an extra projection only
to text does not preserve FashionCLIP's original text/image geometry.

The current evidence points to the compatibility helper applying that extra
projection. Repairing that helper and rebuilding a synchronized query/image
bundle would change the saved embeddings. That is a separate decision from
this task's requirement to preserve and verify the current 240 query vectors.

## Sources

- [Upstream FashionCLIP implementation](https://github.com/patrickjohncyh/fashion-clip/blob/master/fashion_clip/fashion_clip.py)
  maps fashion-clip to patrickjohncyh/fashion-clip and defines preprocessing.
- [Pinned model repository](https://huggingface.co/patrickjohncyh/fashion-clip/tree/7e3ba62ce16b379a1ab479346b66f192e76f51b7)
  supplies the weights and processor.
- [Transformers 5.3.0 CLIP source](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/clip/modeling_clip.py)
  returns projected features in pooler_output; applying text_projection again
  therefore changes them.
