"""Ablation pipeline: freeze candidates -> score once -> select four variants.

Intuition: an ablation removes one stage to expose its contribution. All variants
share filtering, catalog order, image deduplication, and vectors. Only ordering
and the final category-diversity pass differ. Production functions compute hybrid
scores and diversity; the evaluator does not reproduce their formulas.

The snapshot is a plain file: no user DB, network request, or model load is needed.
Random order uses a per-case seed so adding/reordering other cases changes nothing.
"""

import json
import platform
import random
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import numpy as np

from app.services.analysis.tracing import content_hash
from app.services.recommendation.catalog_policy import (
    MALE_GENDER_ALIASES,
    unique_image_outfits,
)
from app.services.recommendation.ranking import (
    _cosine_similarity_search,
    _rank_candidates,
    _select_diverse_top_k,
)
from app.services.recommendation.types import RankedCandidate

from .metrics import ranking_metrics
from .provenance import git_output, source_fingerprint
from .retrieval_schema import (
    RetrievalCase,
    RetrievalProfile,
    RetrievalSnapshot,
    SnapshotOutfit,
)

VARIANTS = ("filtered_random", "vector_only", "hybrid", "hybrid_diverse")


def candidate_pool(
    snapshot: RetrievalSnapshot, case: RetrievalCase
) -> list[SnapshotOutfit]:
    return candidates_for_profile(snapshot.outfits, case.profile)


def candidates_for_profile(
    outfits: Sequence[SnapshotOutfit], profile: RetrievalProfile
) -> list[SnapshotOutfit]:
    """The same eligible pool can be prepared before relevance labels exist."""
    eligible = [
        outfit
        for outfit in outfits
        if outfit.occasion == profile.occasion
        and outfit.gender.lower() in MALE_GENDER_ALIASES
    ]
    return unique_image_outfits(eligible)


def rank_variants(
    pool: list[SnapshotOutfit], case: RetrievalCase, *, seed: int, k: int
) -> dict[str, list[RankedCandidate]]:
    """Compute similarities/hybrid scores once; return selected candidate records."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not pool:
        raise ValueError(f"No eligible candidates for {case.case_id}")
    scores = _cosine_similarity_search(
        np.asarray(case.query_vector, dtype=np.float32),
        np.asarray([outfit.vector for outfit in pool], dtype=np.float32),
    )
    profile = case.profile
    hybrid = _rank_candidates(
        pool,
        scores,
        color_sub_season=profile.color_sub_season,
        body_shape=profile.body_shape,
        occasion=profile.occasion,
        color_confidence=profile.color_confidence,
        body_confidence=profile.body_confidence,
        color_photo_quality_score=profile.color_photo_quality_score,
        body_photo_quality_score=profile.body_photo_quality_score,
    )
    by_id = {candidate.outfit.id: candidate for candidate in hybrid}
    # Start from catalog order so stable ties match production's input order.
    catalog_order = [by_id[outfit.id] for outfit in pool]
    vector_order = sorted(
        catalog_order, key=lambda item: item.retrieval_score, reverse=True
    )
    random_order = catalog_order.copy()
    case_seed = int(content_hash(f"{seed}:{case.case_id}"), 16)
    random.Random(case_seed).shuffle(random_order)
    return {
        "filtered_random": random_order[:k],
        "vector_only": vector_order[:k],
        "hybrid": hybrid[:k],
        "hybrid_diverse": _select_diverse_top_k(hybrid, k),
    }


def eligible_judgments(
    pool: list[SnapshotOutfit], case: RetrievalCase
) -> dict[str, int]:
    """Require a non-empty, fully judged pool before computing any metric."""
    if not pool:
        raise ValueError(f"No eligible candidates for {case.case_id}")
    judgments = {
        str(outfit.id): case.relevance[str(outfit.id)]
        for outfit in pool
        if str(outfit.id) in case.relevance
    }
    if len(judgments) != len(pool):
        raise ValueError(f"Every eligible candidate needs a judgment: {case.case_id}")
    return judgments


def evaluate_snapshot(
    snapshot_path: Path, *, split: str, seed: int = 42, k: int = 6
) -> dict:
    """Fail preflight for unjudged/empty pools; never drop a case from metrics."""
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = RetrievalSnapshot.model_validate_json(snapshot_bytes.decode("utf-8-sig"))
    source_at_start = source_fingerprint()
    selected = [case for case in snapshot.cases if case.split == split]
    if not selected:
        raise ValueError(f"No retrieval cases in split: {split}")
    rows = []
    for case in selected:
        pool = candidate_pool(snapshot, case)
        judgments = eligible_judgments(pool, case)
        variants = rank_variants(pool, case, seed=seed, k=k)
        for variant, candidates in variants.items():
            ids = [str(candidate.outfit.id) for candidate in candidates]
            rows.append(
                {
                    "case_id": case.case_id,
                    "variant": variant,
                    "candidate_ids": [str(outfit.id) for outfit in pool],
                    "eligible_relevance": judgments,
                    "outfit_ids": ids,
                    "retrieval_scores": [
                        candidate.retrieval_score for candidate in candidates
                    ],
                    "hybrid_scores": [
                        candidate.final_score for candidate in candidates
                    ],
                    "reason_tags": [
                        list(candidate.reason_tags) for candidate in candidates
                    ],
                    "metrics": ranking_metrics(ids, judgments, k=k),
                }
            )
    summary = {}
    for variant in VARIANTS:
        metrics = [row["metrics"] for row in rows if row["variant"] == variant]
        defined_ndcg = [
            item["ndcg_at_k"] for item in metrics if item["ndcg_at_k"] is not None
        ]
        summary[variant] = {
            "case_count": len(metrics),
            "precision_at_k": sum(item["precision_at_k"] for item in metrics)
            / len(metrics),
            "hit_at_k": sum(item["hit_at_k"] for item in metrics) / len(metrics),
            "ndcg_at_k": sum(defined_ndcg) / len(defined_ndcg)
            if defined_ndcg
            else None,
            "ndcg_defined_cases": len(defined_ndcg),
        }
    return {
        "metadata": {
            "dataset_id": snapshot.dataset_id,
            "dataset_version": snapshot.version,
            "snapshot_sha256": content_hash(snapshot_bytes),
            "source_sha256": source_at_start,
            "code_commit": git_output("rev-parse", "HEAD"),
            "python_version": platform.python_version(),
            "numpy_version": version("numpy"),
            "synthetic": snapshot.synthetic,
            "split": split,
            "seed": seed,
            "k": k,
            "embedding": snapshot.embedding.model_dump(),
            "query_signature": snapshot.query_signature,
        },
        "interpretation": (
            "Synthetic algorithm check; does not measure fashion recommendation quality."
            if snapshot.synthetic
            else "Descriptive ranking agreement with independent judgments on the frozen snapshot."
        ),
        "summary": summary,
        "per_case": rows,
    }


def write_ablation_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Keep the IDCG judgment pool identical when rescoring exported predictions.
    judgment_dataset = {
        "dataset_id": report["metadata"]["dataset_id"],
        "version": report["metadata"]["dataset_version"],
        "synthetic": report["metadata"]["synthetic"],
        "cases": [
            {"case_id": row["case_id"], "relevance": row["eligible_relevance"]}
            for row in report["per_case"]
            if row["variant"] == VARIANTS[0]
        ],
    }
    (output_dir / "judgments.dataset.json").write_text(
        json.dumps(judgment_dataset, indent=2), encoding="utf-8"
    )
    lines = [
        "# F3 ablation",
        "",
        report["interpretation"],
        "",
        "| Variant | Cases | Precision@k | Hit@k | NDCG@k |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["summary"].items():
        ndcg = metrics["ndcg_at_k"]
        ndcg_text = f"{ndcg:.4f}" if ndcg is not None else "N/A"
        lines.append(
            f"| {name} | {metrics['case_count']} | {metrics['precision_at_k']:.4f} | {metrics['hit_at_k']:.4f} | {ndcg_text} |"
        )
        predictions = [
            {"case_id": row["case_id"], "outfit_ids": row["outfit_ids"]}
            for row in report["per_case"]
            if row["variant"] == name
        ]
        (output_dir / f"{name}.predictions.json").write_text(
            json.dumps(predictions, indent=2), encoding="utf-8"
        )
    lines += [
        "",
        "All variants share the eligible, image-deduplicated pool.",
        "Hybrid scores are diagnostics, not relevance judgments.",
        "Random is one seeded draw; repeated seeds are needed for a robust random baseline.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
