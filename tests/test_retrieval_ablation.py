"""Compare the evaluator with production and hand-known rankings, not itself."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.database import Base
from app.db.models import Outfit
from app.evaluation.retrieval import (
    VARIANTS,
    candidate_pool,
    evaluate_snapshot,
    rank_variants,
    write_ablation_report,
)
from app.evaluation.retrieval_schema import RetrievalSnapshot
from app.services.outfit_service import recommend_outfits
from scripts.evaluate_ranking_exports import evaluate_exports

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "evals/synthetic/retrieval.snapshot.json"


@pytest.fixture
def snapshot():
    return RetrievalSnapshot.model_validate_json(SNAPSHOT.read_text(encoding="utf-8"))


def test_variants_share_filtered_deduplicated_candidates(snapshot):
    case = snapshot.cases[0]
    pool = candidate_pool(snapshot, case)
    assert [outfit.id for outfit in pool] == [1, 2, 3, 4, 5, 6, 7, 8, 12]
    variants = rank_variants(pool, case, seed=42, k=6)
    assert tuple(variants) == VARIANTS
    assert [item.outfit.id for item in variants["vector_only"]] == [1, 2, 3, 4, 5, 6]
    assert [item.outfit.id for item in variants["hybrid"]] != [1, 2, 3, 4, 5, 6]
    for result in variants.values():
        assert len(result) == 6
        assert len({candidate.outfit.image_url for candidate in result}) == 6
        assert {candidate.outfit.id for candidate in result} <= {
            outfit.id for outfit in pool
        }


def test_random_is_stable_per_case_and_does_not_change_global_rng(snapshot):
    import random

    case = snapshot.cases[0]
    pool = candidate_pool(snapshot, case)
    state = random.getstate()
    first = rank_variants(pool, case, seed=42, k=6)
    again = rank_variants(pool, case, seed=42, k=6)
    another_seed = rank_variants(pool, case, seed=43, k=6)

    def ids(variants):
        return [item.outfit.id for item in variants["filtered_random"]]

    assert ids(first) == ids(again)
    assert ids(first) != ids(another_seed)
    assert random.getstate() == state


def test_full_variant_matches_production_filter_scores_and_order(snapshot):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            for record in snapshot.outfits:
                fields = record.model_dump(exclude={"vector"})
                # Fixed test vectors use the existing injected-encoder test hook.
                db.add(
                    Outfit(
                        **fields,
                        vector_blob=np.asarray(
                            record.vector, dtype=np.float32
                        ).tobytes(),
                        embedding_backend="fashion_clip",
                        embedding_model="test-only",
                        embedding_version="test-only",
                    )
                )
            db.commit()
            for case in snapshot.cases[:2]:

                class FixedEncoder:
                    def encode_text(self, _texts):
                        return np.asarray([case.query_vector], dtype=np.float32)

                production = recommend_outfits(
                    db, **case.profile.model_dump(), text_encoder=FixedEncoder()
                )
                selected = rank_variants(
                    candidate_pool(snapshot, case), case, seed=42, k=6
                )["hybrid_diverse"]
                assert [item.id for item in production.outfits] == [
                    item.outfit.id for item in selected
                ]
                assert [item.score for item in production.outfits] == pytest.approx(
                    [round(item.final_score, 4) for item in selected]
                )
                assert [
                    item.retrieval_score for item in production.outfits
                ] == pytest.approx(
                    [round(item.retrieval_score, 4) for item in selected]
                )
    finally:
        engine.dispose()


def test_ties_retain_snapshot_order_and_small_pool_keeps_fixed_precision_denominator(
    snapshot,
):
    case = snapshot.cases[0]
    pool = candidate_pool(snapshot, case)[:3]
    for outfit in pool:
        outfit.vector = case.query_vector.copy()
    selected = rank_variants(pool, case, seed=42, k=6)
    assert [item.outfit.id for item in selected["vector_only"]] == [1, 2, 3]


@pytest.mark.parametrize(
    "mutation",
    [
        "zero",
        "nan",
        "short",
        "fake_identity",
        "signature",
        "query",
        "duplicate_id",
        "leaked_subject",
    ],
)
def test_snapshot_rejects_invalid_inputs(mutation):
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if mutation == "zero":
        payload["outfits"][0]["vector"] = [0] * 512
    elif mutation == "nan":
        payload["cases"][0]["query_vector"][0] = float("nan")
    elif mutation == "short":
        payload["outfits"][0]["vector"] = [1, 0]
    elif mutation == "fake_identity":
        payload["embedding"]["backend"] = "fashion_clip"
    elif mutation == "signature":
        payload["query_signature"] = "stale"
    elif mutation == "query":
        payload["cases"][0]["query_text"] = "changed query"
    elif mutation == "duplicate_id":
        payload["outfits"][1]["id"] = payload["outfits"][0]["id"]
    else:
        payload["cases"][2]["subject_id"] = payload["cases"][0]["subject_id"]
    with pytest.raises(ValueError):
        RetrievalSnapshot.model_validate(payload)


def test_unjudged_candidate_is_an_error_and_outputs_are_not_overwritten(tmp_path):
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    del payload["cases"][0]["relevance"]["1"]
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="judgment"):
        evaluate_snapshot(missing, split="dev")
    report = evaluate_snapshot(SNAPSHOT, split="dev")
    assert {item["case_count"] for item in report["summary"].values()} == {2}
    assert report["metadata"]["synthetic"] is True
    assert len(report["per_case"]) == 8
    with pytest.raises(FileExistsError):
        write_ablation_report(tmp_path, report)


def test_reordered_cases_keep_same_random_results(tmp_path):
    original = evaluate_snapshot(SNAPSHOT, split="dev")
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    payload["cases"].reverse()
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(payload), encoding="utf-8")
    changed = evaluate_snapshot(reordered, split="dev")

    def outcomes(report):
        return {
            (row["case_id"], row["variant"]): row["outfit_ids"]
            for row in report["per_case"]
        }

    assert outcomes(original) == outcomes(changed)


def test_exported_judgments_and_predictions_reproduce_ablation_metrics(tmp_path):
    report = evaluate_snapshot(SNAPSHOT, split="dev")
    output = tmp_path / "report"
    write_ablation_report(output, report)
    for variant in VARIANTS:
        rescored = evaluate_exports(
            output / "judgments.dataset.json", output / f"{variant}.predictions.json"
        )
        assert rescored["metrics"] == report["summary"][variant]


def test_retrieval_cli_imports_no_database_and_needs_no_key(tmp_path):
    environment = os.environ.copy()
    environment.pop("GEMINI_API_KEY", None)
    environment["DATABASE_URL"] = "invalid-url-that-must-never-be-read"
    code = (
        "import sys; from scripts.run_retrieval_ablation import main; main(); "
        "assert 'app.db.database' not in sys.modules; "
        "assert 'app.db.models' not in sys.modules"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--snapshot",
            str(SNAPSHOT),
            "--split",
            "dev",
            "--output-dir",
            str(tmp_path / "ablation"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (tmp_path / "ablation/hybrid_diverse.predictions.json").is_file()
