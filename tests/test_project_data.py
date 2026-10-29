import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_FILE = BASE_DIR / "web" / "data" / "project.json"


def load_project() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def read_summary(rel_path: str) -> dict:
    with (PROJECT_ROOT / rel_path).open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def test_local_models_match_their_summary_csv():
    project = load_project()
    local_keys = {"mobilenetv3", "vgg16_block5", "resnet50", "vgg16_b4b5"}
    for model in project["models"]:
        if model["key"] in local_keys:
            row = read_summary(model["source_csv"])
            assert model["accuracy"] == float(row["Test Accuracy (%)"]), model["key"]
            assert model["precision"] == float(row["Macro Precision"]), model["key"]
            assert model["recall"] == float(row["Macro Recall"]), model["key"]
            assert model["f1"] == float(row["Macro F1-Score"]), model["key"]
            assert model["latency_ms"] == float(row["Latency per Image (ms)"]), model["key"]
            assert model["params"] == int(row["Total Params"]), model["key"]


def test_model_file_size_is_measured_for_local_models():
    """CSV ghi 0 cho hai model VGG16 -> phai do lai tu file .h5."""
    project = load_project()
    local_keys = {"mobilenetv3", "vgg16_block5", "resnet50", "vgg16_b4b5"}
    for model in project["models"]:
        if model["key"] in local_keys:
            assert model["file_mb"] is not None
            assert model["file_mb"] > 0
            row = read_summary(model["source_csv"])
            csv_size = float(row["Model Size (MB)"])
            if csv_size == 0:
                assert model["file_mb"] != 0
            else:
                assert abs(model["file_mb"] - csv_size) <= 0.2, model["key"]


def test_freeze_table_matches_report_source():
    project = load_project()
    accuracies = [row["test_accuracy"] for row in project["freeze"]]
    assert accuracies == [90.50, 93.00, 95.12]

    trainable = [row["trainable_params"] for row in project["freeze"]]
    assert trainable == [2_052, 7_081_476, 12_981_252]
    assert all(row["total_params"] == 14_716_740 for row in project["freeze"])


def test_vgg16_b4b5_matches_summary_csv():
    project = load_project()
    vgg = next(m for m in project["models"] if m["key"] == "vgg16_b4b5")
    row = read_summary(vgg["source_csv"])
    assert vgg["accuracy"] == float(row["Test Accuracy (%)"]) == 95.12
    assert vgg["f1"] == float(row["Macro F1-Score"]) == 0.9502


def test_convnext_tiny_metrics_match_source_csv():
    project = load_project()
    convnext = next(m for m in project["models"] if m["key"] == "convnext_tiny")
    csv_path = PROJECT_ROOT / convnext["source_csv"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    full_tune_row = next(r for r in rows if "Freeze 0%" in r["Strategy"])
    assert convnext["accuracy"] == float(full_tune_row["Accuracy (%)"]) == 90.83
    assert convnext["macro_f1"] == round(float(full_tune_row["Macro F1 (%)"]) / 100, 4) == 0.9075
    assert convnext["trainable_params"] == int(full_tune_row["Trainable Params"].replace(",", "")) == 28_019_556


def test_inception_v3_metrics_match_source_csv():
    project = load_project()
    inc = next(m for m in project["models"] if m["key"] == "inception_v3")
    csv_path = PROJECT_ROOT / inc["source_csv"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    full_row = next(r for r in rows if "Full Backbone" in r["Experiment"])
    assert inc["accuracy"] == float(full_row["Test Accuracy (%)"].rstrip("%")) == 95.0
    assert inc["trainable_params"] == int(full_row["Trainable Params"].replace(",", "")) == 21_759_332
    assert inc["evaluation_type"] == "evaluate_only"
    assert inc["macro_f1"] is None


def test_densenet169_metrics_match_source_csv():
    project = load_project()
    d169 = next(m for m in project["models"] if m["key"] == "densenet169")
    csv_path = PROJECT_ROOT / d169["source_csv"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    full_row = next(r for r in rows if "Full Backbone" in r["Experiment"])
    assert round(d169["accuracy"], 2) == round(float(full_row["Test Accuracy (%)"].rstrip("%")), 2) == 94.32
    assert d169["trainable_params"] == int(full_row["Trainable Params"].replace(",", "")) == 12_756_676
    assert d169["evaluation_type"] == "evaluate_only"
    assert d169["macro_f1"] is None


def test_mobilenet_v2_metrics_match_source_csv():
    project = load_project()
    mv2 = next(m for m in project["models"] if m["key"] == "mobilenet_v2")
    csv_path = PROJECT_ROOT / mv2["source_csv"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    full_row = next(r for r in rows if "Full Backbone" in r["Experiment"])
    assert round(mv2["accuracy"], 2) == round(float(full_row["Test Accuracy (%)"].rstrip("%")), 2) == 93.94
    assert mv2["trainable_params"] == int(full_row["Trainable Params"].replace(",", "")) == 2_194_884
    assert mv2["evaluation_type"] == "evaluate_only"
    assert mv2["macro_f1"] is None


def test_models_without_f1_have_evaluate_only_flag():
    project = load_project()
    for model in project["models"]:
        if model["macro_f1"] is None:
            assert model["evaluation_type"] == "evaluate_only", model["key"]
        else:
            assert model["evaluation_type"] == "full", model["key"]


def test_dataset_split_adds_up():
    project = load_project()
    dataset = project["dataset"]
    assert dataset["train"] + dataset["val"] + dataset["test"] == dataset["total"] == 7_200
    assert dataset["test"] == 1_600
    assert dataset["per_class_test"] * 4 == dataset["test"]


def test_accuracy_is_consistent_with_confusion_matrix_totals():
    """1522/1600 = 95.125% ~ 95.12% (ma tran nham phan trong bao cao)."""
    project = load_project()
    vgg = next(m for m in project["models"] if m["key"] == "vgg16_b4b5")
    correct = round(vgg["accuracy"] * project["dataset"]["test"] / 100)
    assert abs(correct / project["dataset"]["test"] * 100 - vgg["accuracy"]) < 0.01


def test_class_list_is_sorted_like_the_confusion_matrix():
    project = load_project()
    keys = [item["key"] for item in project["dataset"]["classes"]]
    assert keys == ["glioma", "meningioma", "notumor", "pituitary"]


def test_project_contains_all_eight_architectures():
    project = load_project()
    keys = {m["key"] for m in project["models"]}
    expected = {
        "densenet121",
        "vgg16_b4b5",
        "inception_v3",
        "densenet169",
        "mobilenet_v2",
        "resnet50",
        "convnext_tiny",
        "mobilenetv3",
    }
    assert expected.issubset(keys), f"Thieu cac mo hinh: {expected - keys}"


def test_overall_best_model_is_densenet121():
    project = load_project()
    best = max(project["models"], key=lambda item: item["accuracy"])
    assert best["key"] == "densenet121"
    assert best["accuracy"] == 97.29
    assert best["macro_f1"] == 0.9729


def test_project_has_three_research_tracks():
    project = load_project()
    assert "tracks" in project
    tracks = project["tracks"]
    assert set(tracks.keys()) == {"vgg", "resnet", "mobilenet"}
    for key, track in tracks.items():
        assert "name" in track
        assert "lead" in track
        assert "runs" in track
        assert len(track["runs"]) >= 2
        assert "cm_image" in track
        assert (BASE_DIR / "web" / track["cm_image"]).exists(), f"Thieu CM cho track {key}"
        assert (BASE_DIR / "web" / track["xai_image"]).exists(), f"Thieu XAI cho track {key}"


def test_no_student_jargon_in_project_json():
    content = DATA_FILE.read_text(encoding="utf-8")
    assert "toàn nhóm" not in content, "project.json van con chua 'toàn nhóm'"
    assert "toàn bộ đề tài" not in content, "project.json van con chua 'toàn bộ đề tài'"


