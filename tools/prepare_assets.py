"""Chuan bi asset + sinh web/data/project.json cho trang web.

Moi so lieu tren trang duoc doc tu file CSV ket qua cua nhom nen
tests/test_project_data.py kiem tra lai khong khop se bao loi.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
ASSET_DIR = BASE_DIR / "web" / "assets"
DATA_DIR = BASE_DIR / "web" / "data"

MAX_WIDTH = 1400

# (duong dan nguon, duong dan dich trong web/assets)
IMAGE_ASSETS = [
    ("project 1/report_asset/results_vgg16_b4b5/confusion_matrix.png", "results/confusion_matrix_b4b5.png", "plain"),
    ("project 1/report_asset/vgg16_block5/confusion_matrix.png", "results/confusion_matrix_block5.png", "plain"),
    ("project 1/report_asset/resnet/confusion_matrix.png", "results/confusion_matrix_resnet50.png", "plain"),
    ("project 1/report_asset/mobilenet/confusion_matrix.png", "results/confusion_matrix_mobilenetv3.png", "plain"),
    ("project 1/report_asset/results_vgg16_b4b5/labels_vs_counts.png", "dataset/labels_vs_counts.png", "plain"),
    ("project 1/report_asset/results_vgg16_b4b5/model_architecture.png", "method/vgg16_architecture.png", "plain"),
    ("project 1/report_asset/results_vgg16_b4b5/heatmaps/block1_conv1_heatmap.png", "xai/block1_conv1.png", "trim"),
    ("project 1/report_asset/results_vgg16_b4b5/heatmaps/block2_conv2_heatmap.png", "xai/block2_conv2.png", "trim"),
    ("project 1/report_asset/results_vgg16_b4b5/heatmaps/block3_conv3_heatmap.png", "xai/block3_conv3.png", "trim"),
    ("project 1/report_asset/results_vgg16_b4b5/heatmaps/block4_conv3_heatmap.png", "xai/block4_conv3.png", "trim"),
    ("project 1/report_asset/results_vgg16_b4b5/heatmaps/block5_conv3_heatmap.png", "xai/block5_conv3.png", "trim"),
    ("project 1/report_asset/results_vgg16_b4b5/activations/block5_conv3.png", "xai/activation_block5_conv3.png", "trim"),
    ("project 1/report_asset/resnet/heatmaps/conv5_block3_out_heatmap.png", "xai/resnet_conv5_block3.png", "trim"),
    ("project 1/report_asset/mobilenet/heatmaps/expanded_conv_9_project_heatmap.png", "xai/mobilenet_expanded_conv9.png", "trim"),
]

DOCUMENT_ASSETS = [
    ("Bao_Cao_Phan_Loai_U_Nao_Brain_Tumor_v2.docx", "report/bao-cao-brain-tumor-v2.docx"),
]

# key -> (csv ket qua, ten hien thi, file .h5 da fine-tune de do dung luong)
SUMMARY_SOURCES = {
    "mobilenetv3": (
        "project 1/report_asset/mobilenet/mobilenetv3_summary.csv",
        "MobileNetV3-Small",
        "project 1/report_asset/mobilenet/mobilenetv3_finetuned.h5",
    ),
    "vgg16_block5": (
        "project 1/report_asset/vgg16_block5/vgg16_block5_summary.csv",
        "VGG16 — mở riêng Block 5",
        "project 1/report_asset/vgg16_block5/vgg16_block5_finetuned.h5",
    ),
    "resnet50": (
        "project 1/report_asset/resnet/resnet50_summary.csv",
        "ResNet50",
        "project 1/report_asset/resnet/resnet50_finetuned.h5",
    ),
    "vgg16_b4b5": (
        "project 1/report_asset/results_vgg16_b4b5/vgg16_b4b5_summary.csv",
        "VGG16 — mở Block 4 + Block 5",
        "project 1/report_asset/results_vgg16_b4b5/vgg16_block5_finetuned.h5",
    ),
}

MODEL_NOTES = {
    "mobilenetv3": "Mô hình siêu nhẹ làm mốc so sánh: mở toàn bộ khiến mô hình học nhiễu.",
    "vgg16_block5": "Chỉ thích nghi tầng sâu nhất — thêm 2,50 điểm so với đóng băng toàn bộ.",
    "resnet50": "Cân bằng giữa độ chính xác và tốc độ, nhờ skip connection.",
    "vgg16_b4b5": "Mô hình tốt nhất, dùng cho demo trực tiếp.",
}

FREEZE_TABLE = [
    {
        "stage": "Khóa 100% (Feature Extraction)",
        "trainable_params": 2_052,
        "total_params": 14_716_740,
        "test_accuracy": 90.50,
        "note": "Chỉ huấn luyện đầu phân loại, giữ nguyên đặc trưng ImageNet.",
        "source": "create_report.py:279",
    },
    {
        "stage": "Mở riêng Block 5",
        "trainable_params": 7_081_476,
        "total_params": 14_716_740,
        "test_accuracy": 93.00,
        "note": "Thích nghi đặc trưng u não ở tầng sâu nhất.",
        "source": "create_report.py:280 + đo trực tiếp từ file .h5",
    },
    {
        "stage": "Mở Block 4 + Block 5",
        "trainable_params": 12_981_252,
        "total_params": 14_716_740,
        "test_accuracy": 95.12,
        "note": "Mở thêm một tầng, bắt được cả đặc trưng vùng tổn thương.",
        "source": "vgg16_b4b5_summary.csv + đo trực tiếp từ file .h5",
    },
]


def _optimize(source: Path, target: Path, mode: str = "plain", max_width: int = MAX_WIDTH) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if mode == "trim":
            image = _trim_figure(image)
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        image.save(target, format="PNG", optimize=True)


def _trim_figure(image: Image.Image) -> Image.Image:
    """Cắt dải tiêu đề matplotlib và lề trắng, chỉ giữ lại phần nội dung."""
    array = np.asarray(image)
    nonwhite = (array < 245).any(axis=2)
    row_fraction = nonwhite.mean(axis=1)
    dense = np.where(row_fraction > 0.4)[0]
    if dense.size == 0:
        return image
    top = int(dense[0])
    region = nonwhite[top:]
    rows = np.where(region.any(axis=1))[0]
    cols = np.where(region.any(axis=0))[0]
    bottom = top + int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    return image.crop((left, top, right, bottom))


def copy_assets() -> list[str]:
    # Xoa thu muc asset cu de khong con file cu bien doi danh nghiep
    for folder in ("dataset", "method", "results", "xai", "demo"):
        target = ASSET_DIR / folder
        if target.exists():
            shutil.rmtree(target)

    copied: list[str] = []
    for source_rel, target_rel, mode in IMAGE_ASSETS:
        source = PROJECT_ROOT / source_rel
        if not source.exists():
            print(f"  [bo qua] khong tim thay: {source_rel}")
            continue
        _optimize(source, ASSET_DIR / target_rel, mode)
        copied.append(target_rel)

    for source_rel, target_rel in DOCUMENT_ASSETS:
        source = PROJECT_ROOT / source_rel
        if not source.exists():
            print(f"  [bo qua] khong tim thay: {source_rel}")
            continue
        target = ASSET_DIR / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        copied.append(target_rel)

    # Luoi 9 bien the tang cuong (cat tu anh augmentation co truc toa do)
    for tile in sorted((BASE_DIR / "tests" / "fixtures").glob("tile_*.png")):
        target_rel = f"dataset/aug_{tile.stem[-2:]}.png"
        _optimize(tile, ASSET_DIR / target_rel, "plain", max_width=340)
        copied.append(target_rel)

    sample = BASE_DIR / "tests" / "fixtures" / "tile_04.png"
    if sample.exists():
        _optimize(sample, ASSET_DIR / "demo" / "sample_mri.png", "plain", max_width=900)
        copied.append("demo/sample_mri.png")

    return copied


def read_summary(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    return row


FAMILIES = [
    {"id": "all", "name": "Tất cả kiến trúc", "count": 9},
    {"id": "dense", "name": "Dense Connection", "count": 2},
    {"id": "vgg", "name": "Sequential ConvNet", "count": 2},
    {"id": "resnet", "name": "Residual Connection", "count": 1},
    {"id": "inception", "name": "Multi-branch Inception", "count": 1},
    {"id": "convnext", "name": "Modern Pure ConvNet", "count": 1},
    {"id": "mobilenet", "name": "Inverted Residuals", "count": 2},
]


def build_models() -> list[dict]:
    return [
        {
            "key": "densenet121",
            "family": "dense",
            "family_name": "Dense Connection",
            "track": "dense",
            "name": "DenseNet-121 (Full Fine-Tuning)",
            "params": 7219332,
            "trainable_params": 7219332,
            "weights_mb": 28.9,
            "file_mb": None,
            "accuracy": 97.29,
            "precision": 0.9730,
            "recall": 0.9730,
            "macro_f1": 0.9729,
            "f1": 0.9729,
            "latency_ms": None,
            "train_min": 10.60,
            "finetune_min": None,
            "evaluation_type": "full",
            "is_demo_model": False,
            "note": "Top 1 Benchmark. Stratified 5-Fold đạt 92.46% ± 1.14%.",
            "source_file": "project 1/output/Project_AI-20261029T074600Z-1-002/Project_AI/densenet/densenet121/Kqua Kaggel Dense121/Bao_Cao_Thuc_Nghiem_DenseNet121_Brain_Tumor.html",
            "source_csv": "project 1/output/Project_AI-20261029T074600Z-1-002/Project_AI/densenet/densenet121/Kqua Kaggel Dense121/densenet121_kfold_summary.csv"
        },
        {
            "key": "vgg16_b4b5",
            "family": "vgg",
            "family_name": "Sequential ConvNet",
            "track": "vgg",
            "name": "VGG16 — mở Block 4 + Block 5",
            "params": 14716740,
            "trainable_params": 12981252,
            "weights_mb": 58.9,
            "file_mb": 155.3,
            "accuracy": 95.12,
            "precision": 0.9535,
            "recall": 0.9512,
            "macro_f1": 0.9502,
            "f1": 0.9502,
            "latency_ms": 75.42,
            "train_min": 43.55,
            "finetune_min": 29.67,
            "evaluation_type": "full",
            "is_demo_model": True,
            "note": "Mô hình tốt nhất có sẵn file trọng số, tối ưu cho phòng thử nghiệm trực tiếp.",
            "source_csv": "project 1/report_asset/results_vgg16_b4b5/vgg16_b4b5_summary.csv"
        },
        {
            "key": "inception_v3",
            "family": "inception",
            "family_name": "Multi-branch Inception",
            "track": "inception",
            "name": "InceptionV3 (Full Backbone)",
            "params": 21759332,
            "trainable_params": 21759332,
            "weights_mb": 87.0,
            "file_mb": None,
            "accuracy": 95.00,
            "precision": None,
            "recall": None,
            "macro_f1": None,
            "f1": None,
            "latency_ms": None,
            "train_min": 31.28,
            "finetune_min": None,
            "evaluation_type": "evaluate_only",
            "is_demo_model": False,
            "note": "Đo bằng model.evaluate() trên tập kiểm thử (Loss: 0.4110).",
            "source_csv": "project 1/output/Project_AI-20261029T074600Z-1-002/Project_AI/Inception/Inceptionv3/finetuning_comparison_summary.csv"
        },
        {
            "key": "densenet169",
            "family": "dense",
            "family_name": "Dense Connection",
            "track": "dense",
            "name": "DenseNet-169 (Full Backbone)",
            "params": 12756676,
            "trainable_params": 12756676,
            "weights_mb": 51.0,
            "file_mb": None,
            "accuracy": 94.32,
            "precision": None,
            "recall": None,
            "macro_f1": None,
            "f1": None,
            "latency_ms": None,
            "train_min": 19.44,
            "finetune_min": None,
            "evaluation_type": "evaluate_only",
            "is_demo_model": False,
            "note": "Đo bằng model.evaluate() trên tập kiểm thử (Loss: 0.3584).",
            "source_csv": "project 1/output/Project_AI-20261029T074600Z-1-002/Project_AI/densenet/densenet169/Dat/finetuning_comparison_summary.csv"
        },
        {
            "key": "mobilenet_v2",
            "family": "mobilenet",
            "family_name": "Inverted Residuals",
            "track": "mobilenet",
            "name": "MobileNetV2 (Full Backbone)",
            "params": 2194884,
            "trainable_params": 2194884,
            "weights_mb": 8.8,
            "file_mb": None,
            "accuracy": 93.94,
            "precision": None,
            "recall": None,
            "macro_f1": None,
            "f1": None,
            "latency_ms": None,
            "train_min": 28.52,
            "finetune_min": None,
            "evaluation_type": "evaluate_only",
            "is_demo_model": False,
            "note": "Đo bằng model.evaluate() trên tập kiểm thử (Loss: 0.3849).",
            "source_csv": "project 1/output/Project_AI-20261029T074600Z-1-002/Project_AI/mobilenet/mobilenetv2/finetuning_comparison_summary.csv"
        },
        {
            "key": "resnet50",
            "family": "resnet",
            "family_name": "Residual Skip-connection",
            "track": "resnet",
            "name": "ResNet50 (Fine-tuned)",
            "params": 23595908,
            "trainable_params": 23595908,
            "weights_mb": 94.4,
            "file_mb": 270.4,
            "accuracy": 93.00,
            "precision": 0.9324,
            "recall": 0.9300,
            "macro_f1": 0.9286,
            "f1": 0.9286,
            "latency_ms": 57.05,
            "train_min": 18.78,
            "finetune_min": 24.13,
            "evaluation_type": "full",
            "is_demo_model": True,
            "note": "Tốc độ nhanh nhất (57.05 ms) nhờ xử lý song song trên nhân CUDA.",
            "source_csv": "project 1/report_asset/resnet/resnet50_summary.csv"
        },
        {
            "key": "vgg16_block5",
            "family": "vgg",
            "family_name": "Sequential ConvNet",
            "track": "vgg",
            "name": "VGG16 — mở riêng Block 5",
            "params": 14716740,
            "trainable_params": 7081476,
            "weights_mb": 58.9,
            "file_mb": 110.3,
            "accuracy": 93.00,
            "precision": 0.9320,
            "recall": 0.9300,
            "macro_f1": 0.9289,
            "f1": 0.9289,
            "latency_ms": 71.19,
            "train_min": 44.87,
            "finetune_min": 29.51,
            "evaluation_type": "full",
            "is_demo_model": True,
            "note": "Khảo sát đóng băng: chỉ thích nghi tầng sâu nhất, tiết kiệm tham số.",
            "source_csv": "project 1/report_asset/vgg16_block5/vgg16_block5_summary.csv"
        },
        {
            "key": "convnext_tiny",
            "family": "convnext",
            "family_name": "Modern Pure ConvNet",
            "track": "convnext",
            "name": "ConvNeXt-Tiny (Full Fine-Tuning)",
            "params": 28019556,
            "trainable_params": 28019556,
            "weights_mb": 112.1,
            "file_mb": None,
            "accuracy": 90.83,
            "precision": 0.9172,
            "recall": 0.9083,
            "macro_f1": 0.9075,
            "f1": 0.9075,
            "latency_ms": None,
            "train_min": 18.42,
            "finetune_min": None,
            "evaluation_type": "full",
            "is_demo_model": False,
            "note": "Kiến trúc ConvNet hiện đại hóa CVPR 2022 (7x7 Depthwise Conv).",
            "source_csv": "project 1/output/Project_AI-20261029T074600Z-1-002/Project_AI/Ph/brain_tumor_convnext_tiny_kaggle/brain_tumor_convnext_tiny_benchmark-2/metrics/convnext_tiny_ablation_freezing_results.csv"
        },
        {
            "key": "mobilenetv3",
            "family": "mobilenet",
            "family_name": "Inverted Residuals",
            "track": "mobilenet",
            "name": "MobileNetV3-Small",
            "params": 941428,
            "trainable_params": 941428,
            "weights_mb": 3.8,
            "file_mb": 11.3,
            "accuracy": 68.44,
            "precision": 0.7193,
            "recall": 0.6844,
            "macro_f1": 0.6721,
            "f1": 0.6721,
            "latency_ms": 80.78,
            "train_min": 16.57,
            "finetune_min": 9.65,
            "evaluation_type": "full",
            "is_demo_model": False,
            "note": "Mô hình siêu nhẹ: mở toàn bộ 100% gây hiện tượng Overfitting.",
            "source_csv": "project 1/report_asset/mobilenet/mobilenetv3_summary.csv"
        }
    ]


def build_project(copied: list[str]) -> dict:
    return {
        "benchmark": {
            "name": "Brain Tumor MRI Classification Benchmark",
            "task": "4-Class Brain Tumor MRI Classification",
            "evaluation_metric": "Stratified 5-Fold Cross Validation & Test Set Macro F1",
        },
        "dataset": {
            "name": "Brain Tumor MRI Dataset",
            "source": "Kaggle — masoudnickparvar/brain-tumor-mri-dataset",
            "total": 7_200,
            "train": 4_480,
            "val": 1_120,
            "test": 1_600,
            "per_class_train": 1_120,
            "per_class_test": 400,
            "classes": [
                {
                    "key": "glioma",
                    "label": "U thần kinh đệm",
                    "en": "Glioma",
                    "desc": "Phát sinh từ tế bào thần kinh, thường nằm sâu trong chất não và xâm lấn lan — loại u khó phân biệt nhất với mô bình thường.",
                },
                {
                    "key": "meningioma",
                    "label": "U màng não",
                    "en": "Meningioma",
                    "desc": "Phát sinh từ màng bao bọc não, thường nằm ngoài trung tâm và viền rõ nét — mô hình học được điều này từ tầng Block 4 trở đi.",
                },
                {
                    "key": "notumor",
                    "label": "Không có u",
                    "en": "No tumor",
                    "desc": "Ảnh não bình thường, dùng làm lớp đối chứng để mô hình không “chọn bừa” một lớp bệnh.",
                },
                {
                    "key": "pituitary",
                    "label": "U tuyến yên",
                    "en": "Pituitary",
                    "desc": "Khối u ở vùng tuyến yên giữa não, kích thước nhỏ và vị trí cố định — dễ nhận nhất trong bốn lớp.",
                },
            ],
            "preprocessing": "RGB 224×224, scale 1/255, tăng cường: zoom, lật ngang, xoay ±10°, dịch 10%, brightness 0.55–0.9",
        },
        "models": build_models(),
        "families": FAMILIES,
        "freeze": FREEZE_TABLE,
        "tracks": {
            "vgg": {
                "id": "vgg",
                "name": "Họ VGG (Tích chập tuần tự)",
                "lead": "Khảo sát độ sâu tinh chỉnh (Freezing depths) — chứng minh việc mở khóa thêm tầng Block 4+5 (12.98M tham số) giúp mô hình thích nghi sâu các biến dạng ranh giới nhu mô não, nâng Test Accuracy từ 90.50% lên 95.12%.",
                "badge": "Top Accuracy (95.12%)",
                "runs": [
                    {
                        "name": "VGG16 (Khóa 100% — Baseline)",
                        "trainable_params": 2052,
                        "total_params": 14716740,
                        "test_accuracy": 90.50,
                        "gain": None,
                        "note": "Hội tụ an toàn, chỉ huấn luyện đầu phân loại Dense."
                    },
                    {
                        "name": "VGG16 (Mở riêng Block 5)",
                        "trainable_params": 7081476,
                        "total_params": 14716740,
                        "test_accuracy": 93.00,
                        "gain": "+2.50%",
                        "note": "Thích nghi tầng trích xuất đặc trưng ngữ nghĩa cao."
                    },
                    {
                        "name": "VGG16 (Mở Block 4 + Block 5)",
                        "trainable_params": 12981252,
                        "total_params": 14716740,
                        "test_accuracy": 95.12,
                        "gain": "+2.12%",
                        "note": "Mô hình tối ưu nhất cho suy luận trực tiếp, cô lập chuẩn xác ranh giới u.",
                        "is_best": True
                    }
                ],
                "cm_image": "assets/results/confusion_matrix_b4b5.png",
                "xai_image": "assets/xai/block5_conv3.png",
                "xai_text": "Bản đồ kích hoạt từ tầng block5_conv3 cô lập trọn vẹn ranh giới khối u nội sọ."
            },
            "resnet": {
                "id": "resnet",
                "name": "Họ ResNet50 (Kết nối thặng dư)",
                "lead": "Khắc phục triệt tiêu đạo hàm qua 50 tầng bằng cơ chế kết nối tắt (Skip Connection), đạt tốc độ suy luận nhanh nhất (57.05 ms/ảnh trên GPU) nhờ tối ưu song song nhân CUDA.",
                "badge": "Tốc độ nhanh nhất (57.05 ms)",
                "runs": [
                    {
                        "name": "ResNet50 (Khóa thân mạng — Baseline)",
                        "total_params": 23595908,
                        "file_mb": 90.6,
                        "note": "Khóa thân trích xuất đặc trưng ImageNet."
                    },
                    {
                        "name": "ResNet50 (Fine-tuned toàn phần)",
                        "total_params": 23595908,
                        "test_accuracy": 93.00,
                        "macro_f1": 0.9286,
                        "latency_ms": 57.05,
                        "file_mb": 270.4,
                        "note": "Kiến trúc Bottleneck nén-bung kênh, tối ưu suy luận thời gian thực.",
                        "is_fastest": True
                    }
                ],
                "cm_image": "assets/results/confusion_matrix_resnet50.png",
                "xai_image": "assets/xai/resnet_conv5_block3.png",
                "xai_text": "Khối conv5_block3_out tập trung cao độ vào đường viền khối u thặng dư."
            },
            "mobilenet": {
                "id": "mobilenet",
                "name": "Họ MobileNetV3 (Tối ưu thiết bị nhúng)",
                "lead": "Khảo sát kiến trúc siêu nhẹ (<1M tham số) với phép tích chập phân tách chiều sâu (Depthwise Separable). Bài học điển hình về hiện tượng quá khớp (Overfitting): mở khóa 100% khiến Test Accuracy tụt từ 83.84% xuống 68.44%.",
                "badge": "Siêu nhẹ (11.3 MB)",
                "runs": [
                    {
                        "name": "MobileNetV3-Small (100% Khóa — Baseline)",
                        "total_params": 941428,
                        "val_accuracy": 83.84,
                        "file_mb": 4.1,
                        "note": "Hội tụ an toàn với đặc trưng ImageNet, tránh phá hủy phân phối."
                    },
                    {
                        "name": "MobileNetV3-Small (100% Mở khóa Fine-tune)",
                        "total_params": 941428,
                        "test_accuracy": 68.44,
                        "macro_f1": 0.6721,
                        "latency_ms": 80.78,
                        "file_mb": 11.3,
                        "note": "Hiện tượng Overfitting: 210/400 ca Meningioma bị đoán nhầm sang Glioma."
                    }
                ],
                "cm_image": "assets/results/confusion_matrix_mobilenetv3.png",
                "xai_image": "assets/xai/mobilenet_expanded_conv9.png",
                "xai_text": "Khối chú ý kênh Squeeze-and-Excitation (SE) tự động lọc bỏ nền đen sọ não."
            }
        },
        "xai_layers": [
            {"file": "assets/xai/block1_conv1.png", "layer": "block1_conv1", "title": "Block 1 — cạnh & kết cấu thô", "text": "Tầng đầu chỉ nhìn thấy cạnh sáng tối và đường viền."},
            {"file": "assets/xai/block2_conv2.png", "layer": "block2_conv2", "title": "Block 2 — kết cấu bề mặt", "text": "Các đường nét, vân và kết cấu mô bắt đầu hình thành."},
            {"file": "assets/xai/block3_conv3.png", "layer": "block3_conv3", "title": "Block 3 — mô hình cục bộ", "text": "Mạng bắt đầu gom nhóm đặc trưng theo từng vùng nhỏ."},
            {"file": "assets/xai/block4_conv3.png", "layer": "block4_conv3", "title": "Block 4 — vùng tổn thương", "text": "Đặc trưng bắt đầu tập trung quanh vùng bất thường."},
            {"file": "assets/xai/block5_conv3.png", "layer": "block5_conv3", "title": "Block 5 — quyết định phân loại", "text": "Tầng được mở khóa fine-tune, trực tiếp quyết định nhãn u."},
        ],
        "assets": sorted(copied),
        "notes": [
            "Đường cong huấn luyện không đưa lên trang này: các file CSV lịch sử trong thư mục kết quả có tên bị hoán đổi (baseline ↔ finetuned), nên không thể gắn nhãn chắc chắn cho từng đường cong.",
            "Cột “Model Size” trong CSV gốc của hai mô hình VGG16 bị ghi 0 — trang này dùng dung lượng file .h5 đo trực tiếp.",
            "Bản đồ nhiệt là activation map (trung bình activation của block5_conv3), không phải Grad-CAM.",
            "Kết quả chỉ phục vụ học thuật, không dùng để chẩn đoán lâm sàng.",
        ],
    }


def main() -> None:
    print("  [1/2] Chep bi asset ...")
    copied = copy_assets()
    print(f"        {len(copied)} file da chep")

    print("  [2/2] Sinh web/data/project.json ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    project = build_project(copied)
    (DATA_DIR / "project.json").write_text(
        json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for model in project["models"]:
        f1_str = f"{model['f1']:.4f}" if model.get('f1') is not None else "   —  "
        print(
            f"        {model['name']:32s} acc={model['accuracy']:5.2f} "
            f"f1={f1_str} file={model['file_mb']} MiB"
        )


if __name__ == "__main__":
    main()
