import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import dataset_manager


def test_upload_csv_returns_report_and_preview(monkeypatch):
    base_dir = Path(__file__).resolve().parent / "_api_tmp" / uuid4().hex
    upload_dir = base_dir / "uploads"
    output_dir = base_dir / "output"
    upload_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(settings, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(dataset_manager, "REGISTRY_PATH", str(upload_dir / "registry.json"))

    client = TestClient(app)
    csv_content = (
        "ticket_id,stars,verbatim\n"
        "1,5,The product works beautifully and setup was easy.\n"
        "2,2,Checkout was slow and the page timed out twice.\n"
        "3,4,Support answered quickly and fixed the issue.\n"
        "4,4,Support answered quickly and fixed the issue.\n"
    )

    upload_response = client.post(
        "/api/upload",
        files={"file": ("feedback.csv", csv_content, "text/csv")},
    )

    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["dataset_id"]
    assert payload["report"]["dataset_id"] == payload["dataset_id"]
    assert payload["report"]["success"] is True
    assert payload["report"]["text_column"]["column_name"] == "verbatim"
    assert payload["report"]["text_column"]["method"] == "local_pipeline"
    assert payload["report"]["stats"]["clean_count"] == 4
    assert payload["report"]["stats"]["duplicate_count"] == 1
    assert all("row_indices" not in issue for issue in payload["report"]["issues"])

    preview_response = client.get(f"/api/datasets/{payload['dataset_id']}/preview?limit=2")
    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["showing"] == 2
    assert "original_row_index" in preview["columns"]
    assert "duplicate_frequency" in preview["columns"]
    assert preview["rows"][0]["verbatim"].startswith("The product works")

    datasets_response = client.get("/api/datasets")
    assert datasets_response.status_code == 200
    dataset = datasets_response.json()["datasets"][0]
    assert dataset["id"] == payload["dataset_id"]
    assert dataset["status"] == "ingested"
    assert Path(dataset["clean_csv_path"]).exists()
    assert Path(dataset["report_json_path"]).exists()

    clean_df = pd.read_csv(dataset["clean_csv_path"])
    duplicate_rows = clean_df[clean_df["verbatim"] == "Support answered quickly and fixed the issue."]
    assert duplicate_rows["duplicate_frequency"].tolist() == [2, 2]

    with open(dataset["report_json_path"], "r") as f:
        saved_report = json.load(f)
    assert saved_report["success"] is True
    assert saved_report["dataset_id"] == payload["dataset_id"]
    assert "row_indices" not in saved_report["issues"][0]
    assert "row_indices_sample" in saved_report["issues"][0]

    analysis_response = client.post(f"/api/datasets/{payload['dataset_id']}/analyze")
    assert analysis_response.status_code == 200
    analysis = analysis_response.json()
    assert analysis["cleaning_report"]["cleaning_status"] in {"cleaned", "already_clean"}
    assert analysis["clean_dataset_path"]
    assert len(analysis["charts"]) <= 8

    download_response = client.get(f"/api/download/clean-dataset/{payload['dataset_id']}")
    assert download_response.status_code == 200
    assert "clean_feedback.csv" in download_response.headers["content-disposition"]

    downloaded_df = pd.read_csv(BytesIO(download_response.content))
    assert downloaded_df.columns.tolist() == ["ticket_id", "stars", "verbatim"]
