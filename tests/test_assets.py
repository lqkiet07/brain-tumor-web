import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"


def load_project() -> dict:
    return json.loads((WEB_DIR / "data" / "project.json").read_text(encoding="utf-8"))


def test_every_listed_asset_exists():
    project = load_project()
    assert project["assets"], "Danh sach asset rong"
    for relative in project["assets"]:
        target = WEB_DIR / "assets" / relative
        assert target.exists(), f"Thieu asset: {relative}"
        assert target.stat().st_size > 0, f"Asset rong: {relative}"


def test_xai_images_exist():
    project = load_project()
    for layer in project["xai_layers"]:
        target = WEB_DIR / layer["file"]
        assert target.exists(), f"Thieu anh XAI: {layer['file']}"


def test_demo_sample_and_report_exist():
    assert (WEB_DIR / "assets" / "demo" / "sample_mri.png").exists()
    assert (WEB_DIR / "assets" / "report" / "bao-cao-brain-tumor-v2.docx").exists()


def test_frontend_files_exist():
    for relative in ("index.html", "css/style.css", "js/app.js", "js/demo.js"):
        assert (WEB_DIR / relative).exists(), f"Thieu file frontend: {relative}"
