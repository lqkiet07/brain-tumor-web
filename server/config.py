from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
WEB_DIR = BASE_DIR / "web"
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"

# Thứ tự lớp cố định của mô hình (flow_from_directory sắp xếp alphabet):
# glioma, meningioma, notumor, pituitary — khớp trục ma trận nhầm lẫn của báo cáo.
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")

CLASS_LABELS_VI = {
    "glioma": "U thần kinh đệm (Glioma)",
    "meningioma": "U màng não (Meningioma)",
    "notumor": "Không có u (No tumor)",
    "pituitary": "U tuyến yên (Pituitary)",
}

CLASS_SHORT_VI = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary",
}

IMG_SIZE = 224
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
OVERLAY_ALPHA = 0.4

MODEL_REGISTRY = {
    "vgg16_b4b5": {
        "id": "vgg16_b4b5",
        "name": "VGG16 (Mở Block 4 + 5)",
        "track": "vgg",
        "researcher": "Lê Quốc Kiệt",
        "path": BASE_DIR / "models" / "vgg16_b4b5.h5",
        "preprocess": "scale_255",
        "heatmap_layer": "block5_conv3",
        "accuracy": 95.12,
        "f1": 0.9502,
        "reported_latency_ms": 75.42,
        "total_params": 14_716_740,
        "trainable_params": 12_981_252,
        "badge": "Top Accuracy (95.12%)",
        "live_supported": True,
    },
    "resnet50": {
        "id": "resnet50",
        "name": "ResNet50 (Fine-tuned)",
        "track": "resnet",
        "researcher": "Thành viên 2 (ResNet Track)",
        "path": BASE_DIR / "models" / "resnet50.h5",
        "preprocess": "resnet_caffe",
        "heatmap_layer": "conv5_block3_out",
        "accuracy": 93.00,
        "f1": 0.9286,
        "reported_latency_ms": 57.05,
        "total_params": 23_595_908,
        "trainable_params": 23_595_908,
        "badge": "Nhanh nhất (57.05 ms)",
        "live_supported": True,
    },
    "vgg16_block5": {
        "id": "vgg16_block5",
        "name": "VGG16 (Mở riêng Block 5)",
        "track": "vgg",
        "researcher": "Lê Quốc Kiệt",
        "path": BASE_DIR / "models" / "vgg16_block5.h5",
        "preprocess": "scale_255",
        "heatmap_layer": "block5_conv3",
        "accuracy": 93.00,
        "f1": 0.9289,
        "reported_latency_ms": 71.19,
        "total_params": 14_716_740,
        "trainable_params": 7_081_476,
        "badge": "Ablation Run",
        "live_supported": True,
    },
}

DEFAULT_MODEL_ID = "vgg16_b4b5"
