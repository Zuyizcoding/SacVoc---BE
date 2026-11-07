"""Offline end-to-end checks: SDK -> workflow -> evidence -> metrics.

All images are generated; responses are scripted at the HTTP boundary. A SQLite
sentinel verifies that evaluation leaves user data unchanged. A fresh subprocess
also verifies that running the CLI does not even import the database layer.
"""

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from app.evaluation.runner import run_dataset
from app.evaluation.schemas import AnalysisDataset, RankingDataset
from scripts.evaluate_ranking_exports import evaluate_exports

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals/synthetic/analysis.dataset.json"


@pytest.fixture(autouse=True)
def no_real_wait(monkeypatch):
    async def sleep(_seconds):
        pass

    monkeypatch.setattr("app.services.gemini_client.asyncio.sleep", sleep)


def test_synthetic_runner_keeps_all_cases_and_exposes_repair_and_failures(tmp_path):
    output = tmp_path / "run"
    report = asyncio.run(
        run_dataset(DATASET, output, provider="synthetic", split="dev")
    )
    rows = [
        json.loads(line)
        for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["case_id"]: row for row in rows}
    assert len(rows) == 10
    assert report["operations"]["case_count"] == 10
    assert report["operations"]["final_success_rate"] == 0.5
    assert report["operations"]["repair_rate"] == 0.3
    assert report["metadata"]["synthetic"] is True
    assert "not AI quality" in report["interpretation"]
    assert report["quality"]["color"]["primary_season"]["case_count"] == 5
    assert report["quality"]["color"]["primary_season"]["accuracy"] == 0.4
    repair = by_id["color-repair"]
    assert [call["stage"] for call in repair["trace"]["calls"]] == ["initial", "repair"]
    assert repair["trace"]["calls"][0]["validated_result"]["advice"] is None
    # Real preprocessing can append a quality warning to the scripted advice.
    assert repair["final_result"]["advice"].startswith("Synthetic advice.")
    assert by_id["color-overload"]["final_result"] is None
    assert by_id["body-incomplete"]["error"]["code"] == "AI_INCOMPLETE"
    assert by_id["color-safety"]["error"]["code"] == "AI_INCOMPLETE"
    assert (
        by_id["color-safety"]["trace"]["calls"][0]["attempts"][0]["error_type"]
        == "GeminiSafetyBlockedError"
    )
    assert by_id["body-clothing"]["final_result"]["confidence"] == 0.45
    assert len(by_id["color-malformed"]["trace"]["calls"][0]["attempts"]) == 2
    assert (output / "run.json").is_file()
    assert (output / "cases.csv").is_file()
    assert (output / "report.md").is_file()
    assert "GEMINI_API_KEY" not in (output / "run.json").read_text(encoding="utf-8")
    assert len(report["metadata"]["source_sha256"]) == 64
    assert len(repair["trace"]["processed_image_sha256"]) == 64


def test_run_directory_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        asyncio.run(run_dataset(DATASET, tmp_path, provider="synthetic", split="dev"))


@pytest.mark.parametrize("provider,split", [("gemini", "dev"), ("synthetic", "test")])
def test_invalid_selection_fails_before_creating_output(tmp_path, provider, split):
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        asyncio.run(run_dataset(DATASET, output, provider=provider, split=split))
    assert not output.exists()


def test_subjects_cannot_leak_across_splits_or_have_duplicate_case_ids():
    manifest = json.loads(DATASET.read_text(encoding="utf-8"))
    manifest["cases"][1]["subject_id"] = manifest["cases"][0]["subject_id"]
    manifest["cases"][1]["split"] = "test"
    with pytest.raises(ValueError, match="Subject appears"):
        AnalysisDataset.model_validate(manifest)
    manifest["cases"][1]["split"] = "dev"
    manifest["cases"][1]["case_id"] = manifest["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="Duplicate case_id"):
        AnalysisDataset.model_validate(manifest)


def test_human_dataset_requires_consent_and_cannot_embed_fake_replies():
    manifest = json.loads(DATASET.read_text(encoding="utf-8"))
    manifest.update(synthetic=False, reference_kind="human_consensus")
    with pytest.raises(ValueError, match="consent"):
        AnalysisDataset.model_validate(manifest)
    case = manifest["cases"][0]
    case.update(consent_confirmed=True, image_path="private/photo.jpg")
    manifest["cases"] = [case]
    with pytest.raises(ValueError, match="synthetic replies"):
        AnalysisDataset.model_validate(manifest)


def test_reference_labels_must_match_taxonomy_and_each_other():
    manifest = json.loads(DATASET.read_text(encoding="utf-8"))
    manifest["cases"][0]["reference"]["primary_season"] = "Winter"
    with pytest.raises(ValueError, match="agree"):
        AnalysisDataset.model_validate(manifest)


def test_ranking_export_missing_predictions_and_undefined_ndcg(tmp_path):
    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": "unit-test",
                "version": "1",
                "synthetic": True,
                "cases": [
                    {"case_id": "missing", "relevance": {"one": 1}},
                    {"case_id": "none-relevant", "relevance": {"one": 0}},
                ],
            }
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        '[{"case_id":"none-relevant","outfit_ids":["one"]}]', encoding="utf-8"
    )
    report = evaluate_exports(path, predictions)
    assert report["metrics"]["case_count"] == 2
    assert report["metrics"]["precision_at_k"] == 0
    assert report["metrics"]["ndcg_at_k"] == 0
    assert report["metrics"]["ndcg_defined_cases"] == 1
    assert report["per_case"][0]["missing_prediction"] is True


def test_cli_needs_no_key_and_never_imports_database(tmp_path):
    database = tmp_path / "user.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('keep me')")
    original = database.read_bytes()
    # One complete case is enough for this subprocess boundary check.
    manifest = deepcopy(json.loads(DATASET.read_text(encoding="utf-8")))
    manifest["cases"] = manifest["cases"][:1]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(manifest), encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("GEMINI_API_KEY", None)
    environment["DATABASE_URL"] = f"sqlite:///{database.as_posix()}"
    code = (
        "import sys; from scripts.run_analysis_evaluation import main; main(); "
        "assert 'app.db.database' not in sys.modules; "
        "assert 'app.db.crud' not in sys.modules"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--dataset",
            str(dataset),
            "--provider",
            "synthetic",
            "--split",
            "dev",
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert database.read_bytes() == original


def test_ranking_dataset_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        RankingDataset.model_validate(
            {
                "dataset_id": "test",
                "version": "1",
                "synthetic": True,
                "cases": [{"case_id": "same", "relevance": {"one": 1}}] * 2,
            }
        )
