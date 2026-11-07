# Evaluation: intuition, schema, pipeline, and commands

For input readiness, read-only F3 snapshot export and a separate CPU PyTorch
learning notebook, continue with [data-preparation.md](data-preparation.md).

## 1. Start with the question

A passing unit test proves that our software follows a rule. It does not prove
that a color season, body shape, or outfit recommendation is correct. Evaluation
compares model/pipeline outputs with independently prepared reference judgments.

There are three execution modes in
[workflows.py](../app/services/analysis/workflows.py):

| Mode | AI call | Availability fallback | Writes history |
| --- | --- | --- | --- |
| live | Yes | Existing demo fallback on overload/timeout | Only through the API wrappers |
| demo | No | Explicit canned output | No |
| evaluation | Yes | Disabled; failure stays a failure | No |

The runner always uses evaluation mode. Its provider is a separate choice:
Gemini makes real API requests; synthetic intercepts HTTP inside the real SDK.
Synthetic runs check software plumbing. They cannot measure AI quality, production
latency, quota, or cost. A synthetic result may still contain analysis_source=live
because it travels through the inference workflow; each evidence row and the run
metadata explicitly identify the synthetic provider.

Evaluation imports neither the endpoint nor database persistence layer. The
existing API still saves demo fallback into the device's history as before.

## 2. Read the implementation in this order

1. [schemas.py](../app/evaluation/schemas.py): what constitutes a valid dataset.
2. [runner.py](../app/evaluation/runner.py): selection, metadata, case execution.
3. [workflows.py](../app/services/analysis/workflows.py): shared business pipeline.
4. [gemini_client.py](../app/services/gemini_client.py): HTTP attempts and validation.
5. [tracing.py](../app/services/analysis/tracing.py): evidence attached to one case.
6. [metrics.py](../app/evaluation/metrics.py): explicit scoring formulas.
7. [reporting.py](../app/evaluation/reporting.py): aggregate and export results.

```mermaid
sequenceDiagram
    participant CLI
    participant Runner
    participant Workflow
    participant Client as GeminiClient
    participant SDK as SDK / provider
    CLI->>Runner: manifest + provider + split + new output directory
    Runner->>Runner: validate cases, subject split, metadata
    loop every selected case
        Runner->>Workflow: analyze_image(mode=evaluation)
        Workflow->>Workflow: preprocess image and quality report
        Workflow->>Client: initial logical call
        loop bounded attempts
            Client->>SDK: image + prompts + Pydantic schema
            SDK-->>Client: raw reply or transport error
            Client->>Client: capture usage; parse and validate
        end
        opt missing business details
            Workflow->>Client: repair logical call
        end
        Workflow-->>Runner: final model or failure
        Runner->>Runner: append and flush cases.jsonl
    end
    Runner-->>CLI: JSON / Markdown / CSV reports
```

A schema is a contract about fields and types. It cannot verify visual truth.
ColorInferenceResponse and BodyGeminiResponse accept a minimal conclusion so the
workflow can repair missing details. ColorApiResponse and BodyApiResponse describe
the final UI result. The color workflow also supplies lookup values and, when
needed, confidence/text defaults. Therefore the report retains the initial reply,
each repair, and the final result instead of evaluating the final object alone.

## 3. Install and run the offline example

Use Python 3.12+ from the backend directory. No Flutter SDK or FashionCLIP package
is needed for this work.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest==8.3.4 ruff==0.9.4

python -m scripts.run_analysis_evaluation --dataset evals/synthetic/analysis.dataset.json --provider synthetic --split dev --output-dir build/evaluation/smoke-001
```

No API key, internet access, real photos, or user DB is needed to run that example.
The SDK and other packages must already be installed. Use a new output directory
for each run: the runner refuses to overwrite existing experiments.

The committed example contains ten scripted cases: success, missing-field repair,
malformed output, overload, clothing best effort, incomplete details, rejection,
and safety blocking. Some cases deliberately fail.

Output files:

| File | Purpose |
| --- | --- |
| run.json | Metadata written before inference, available even if execution is interrupted |
| cases.jsonl | One JSON object per completed case; flushed immediately |
| cases.csv | Compact case status, labels, call counts, and latency |
| report.json | Aggregates, class support, confusion matrices, token observations |
| report.md | Human-readable summary and interpretation limits |

Evidence includes original/processed image hashes, photo quality, exact prompts,
prompt/schema hashes, schema name, per-attempt raw text, normalized call results,
retry delays, error categories, returned model version, observed token usage,
and final result. It does not store image bytes, device IDs, or API keys.

Metadata includes manifest hash, dataset/version, split, requested model, SDK and
runtime versions, Git commit/dirty status, source/prompt bundle hashes, timeout,
attempt budget, thinking budget, safety settings, and preprocessing size limit.
A model alias is not an immutable version; inspect returned model_version too.

Raw model text and repair prompts may contain personal observations. Keep real
datasets and run artifacts private. Local inputs belong in ignored evals/local/
and outputs in ignored build/evaluation/. Only synthetic examples are committed.

## 4. Prepare human reference data

Copy [the template](../evals/templates/analysis.dataset.example.json) to
evals/local/analysis.dataset.json. The template deliberately has consent=false and
placeholder image paths; it must be completed before a live run.

- dataset_id and version identify an immutable dataset snapshot.
- synthetic must be false; reference_kind must be human_consensus.
- annotation_protocol records how raters reached the reference judgments.
- case_id is unique; subject_id is a pseudonym, not a person's name.
- split is dev or test. All images of one person must belong to the same split,
  including their color and body photos. The validator rejects subject leakage.
- image_path is absolute or relative to the manifest's directory.
- consent_confirmed must be true after permission for this use is obtained.
- Color references require primary_season; sub_season is optional and must agree.
- Body references require a canonical shape. Aliases are rejected in references
  to catch annotation mistakes; application inference retains its legacy aliases.

Prepare independent judgments before viewing model predictions. Prefer two raters,
record disagreements separately, and resolve them using a written rubric. Do not
copy model outputs into the reference column. For a small pilot, use four primary
seasons as the main F1 target; twelve sub-seasons need substantially more coverage.
These subjective annotations support agreement claims, not clinical measurements.

Use dev for prompt iteration. Freeze dataset, configuration, and prompts before
using test for a final comparison. The validator detects structural leakage, but
cannot verify consent or whether human annotation was actually independent.

Configure GEMINI_API_KEY securely in the environment or .env, then run:

```powershell
python -m scripts.run_analysis_evaluation --dataset evals/local/analysis.dataset.json --provider gemini --split dev --output-dir build/evaluation/prompt-a-dev-001
```

A real run sends the selected images to Gemini and may incur charges. Model
availability/quota must be verified with your own project. The current model ID
is retained as configuration, not asserted to be available for every key.

## 5. Understand the metrics

Classification reports each taxonomy class's support, precision, recall, F1 and
a confusion matrix whose rows are reference labels and columns are predictions.
Unexpected predictions have their own columns; failures appear as <missing>.

- Accuracy/agreement = correct labels / all cases with that reference label field.
- Coverage = returned labels / those same cases. Failures remain in the denominator.
- Macro F1 = the arithmetic mean of class F1 over the complete declared taxonomy:
  4 primary seasons, 12 sub-seasons, or 5 shapes. Zero-support classes contribute
  zero. Read support alongside macro F1, especially for small pilots.
- Sub-season metrics use only cases that have a sub-season reference.
- Empty label subsets have undefined metrics (null), not perfect scores.
- First-pass schema validity considers the first HTTP attempt of the initial
  logical call; cases that never reach the provider count as unsuccessful.
- Schema-valid-reply rate is conditional on receiving a textual reply.
- Repair rate counts cases that enter the completion/repair stage.
  Clothing best effort is a separate rate.
- Final success means the business pipeline returned its API model; it does not
  mean the visual conclusion matches the reference label.
- p50/p95 include failures, preprocessing, validation, retries and backoff.
  They describe this sample; a small pilot does not establish a production SLO.

Token totals include only usage reported by the provider. Failed requests can
have unreported usage. No currency cost is inferred from missing usage or from
a hard-coded price. Provider defaults such as temperature are recorded as such;
for a fair A/B comparison, keep configuration and dataset unchanged.

## 6. Score F3 ranking exports

The new scorer reads ranking exports and independent item-level judgments. It
does not execute or change the recommender. The new snapshot-based baselines and
ablation runner, paired A/B comparison, and CI are explained in
[experiments.md](experiments.md).

```powershell
python -m scripts.evaluate_ranking_exports --dataset evals/synthetic/ranking.dataset.json --predictions evals/synthetic/ranking.predictions.json --top-k 6 --output build/evaluation/ranking-smoke/report.json
```

Predictions contain case_id and an ordered list of string outfit_ids. Each query's
reference maps outfit ID to grade: 0 irrelevant, 1 weakly relevant, 2 relevant,
3 strongly relevant. Grade > 0 counts as a binary hit. Judge a shared candidate
pool independently of the system's own season/body tags and numeric match score.

Precision@k = number of positive items / k. Empty slots count as misses.
Hit@k = 1 if any returned item is positive, else 0.
NDCG uses linear gain: sum(grade / log2(rank + 1)), divided by the ideal order of
the same judgment pool. When all grades are zero it is undefined (null); the
aggregate reports how many cases have defined NDCG. A missing prediction with
positive judgments earns zero. Duplicate or unjudged IDs inside top-k are errors,
not extra credit or assumed irrelevance.

The old evaluate_f3.py checks name/category expectations against a DB. Its match
scores and name-hit rate are not independent relevance metrics. The old
evaluate_analysis_exports.py retains its flat export CLI for compatibility;
the new runner is the reproducible pipeline evaluation entry point.

## 7. SDK and reliability decisions

The integration uses google-genai==1.56.0, pinned and tested with the existing
Pydantic version. It is a deliberate compatible pin, not a claim to be the latest
release. The old google-generativeai adapter and google.api_core imports are gone.

- Timeout settings remain seconds; HttpOptions receives milliseconds.
- HttpRetryOptions(attempts=1) prevents SDK retries from multiplying our loop.
- GEMINI_MAX_RETRIES keeps its old name and means total attempts per logical call.
- Repair/clothing passes are additional logical calls, each with that budget.
- Invalid JSON/schema retries do not open the circuit breaker.
- Infrastructure failures count toward the breaker; busy and quota responses
  retain their existing distinct stop/retry behavior and backoff.
- The breaker remains shared across cases in a real run, matching one API process.
- Safety blocks are explicit, non-retryable failures; they do not become demos.
- SDK resources are closed at application shutdown or the end of a runner.
- The SDK now receives nested/nullable Pydantic schemas directly and applies the
  configured thinking budget. The legacy adapter could silently omit constraints
  or thinking configuration. Live output equivalence therefore requires a real
  benchmark; mock regression tests prove control flow, not identical generations.

GEMINI_TIMEOUT is per provider attempt. It is not a whole-analysis deadline:
repair, best effort and backoff can exceed the Flutter receive timeout. The sync
SDK runs in a thread; cancelling an awaiting coroutine does not kill that thread.
An end-to-end deadline policy would be a separate behavioral change.

References: [Google SDK migration](https://ai.google.dev/gemini-api/docs/migrate),
[pinned SDK implementation](https://github.com/googleapis/python-genai/tree/v1.56.0),
[structured output](https://ai.google.dev/gemini-api/docs/structured-output).

## 8. Verify changes and investigate a case

```powershell
python -m pytest -q
ruff check app scripts tests
ruff format --check app scripts tests
```

Tests use fake HTTP/AI, generated images, deterministic labels/vectors, and
temporary SQLite. They do not call Gemini, download models, or mutate user data.

When a case fails, inspect its error type/code, initial raw text and schema_valid,
then the stage sequence and retry delays. Finally compare validated_result with
final_result: defaults and quality caps explain many differences. For an A/B
experiment, keep both complete run directories and compare per-case outcomes,
including failures. Do not select only successful cases for a quality claim.
