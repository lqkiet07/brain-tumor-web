<div align="center">

# Brain Tumor MRI Classification Benchmark & Diagnostic Studio

An offline-first, explainable deep learning platform benchmarking **8 CNN architectures across 6 paradigms** on 7,200 brain MRI scans, coupled with an interactive local real-time inference studio.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/framework-TensorFlow%202-FF6F00.svg)](https://tensorflow.org)
[![Tests](https://img.shields.io/badge/tests-61%20passed-success.svg)](./tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Offline Guarantee](https://img.shields.io/badge/network-100%25%20offline-darkgreen.svg)](#offline-first-architecture)

[Key Features](#key-features) • [Benchmark Leaderboard](#benchmark-leaderboard) • [Quickstart](#quickstart) • [Project Structure](#project-structure) • [Explainable AI](#explainable-ai-xai) • [Verification](#testing--verification) • [Citation](#citation--academic-integrity)

</div>

---

## Overview

This repository provides an end-to-end experimental benchmark and local diagnostic web application for 4-class intracranial tumor classification from axial T1/T2-weighted MRI scans:
- **Glioma** (infiltrative glial tissue malignancy)
- **Meningioma** (extra-axial dural sheath neoplasms)
- **Pituitary Adenoma** (sellar region masses)
- **No Tumor** (healthy anatomical baseline control)

The system is engineered with an **offline-first, zero-telemetry architecture**: all deep learning model weights, inference pipelines, and frontend assets (including web fonts) are executed locally on `127.0.0.1` without sending patient scans to any third-party cloud service.

---

## Key Features

- **Multi-Paradigm Comparative Study:** Direct comparative evaluation of 8 distinct architectures spanning classical Sequential ConvNets (VGG), Residual Learning (ResNet), Multi-branch Kernels (Inception), Dense Connectivity (DenseNet), Depthwise Mobile Networks (MobileNetV2/V3), and 2020s Modernized Pure ConvNets (ConvNeXt-Tiny).
- **Interactive Radiology Studio:** Drag-and-drop web UI powered by FastAPI and vanilla JavaScript with dynamic model switching between embedded weights (`vgg16_b4b5`, `resnet50`, `vgg16_block5`).
- **Explainable AI (XAI):** Real-time activation mapping extracted directly from the final convolutional blocks (`block5_conv3`, `conv5_block3_out`) with interactive alpha-blended jet overlays ($0.6 \cdot I_{\text{orig}} + 0.4 \cdot I_{\text{jet}}$).
- **Google-Calm Frontend System:** Bespoke typography and design system adhering to WCAG AA contrast standards ($\ge 4.5:1$), native tabular numerics (`tabular-nums`), responsive collinear tables for mobile viewports, and zero marketing clutter.
- **Strict Verification Quality Gate:** Comprehensive automated test suite comprising 61 unit and contract tests verifying data integrity, model weight topologies, and endpoint contracts.

---

## Benchmark Leaderboard

All metrics reflect evaluation on an isolated holdout test split of **1,600 MRI scans** (400 per diagnostic class) balanced across all four pathologies:

| Rank | Architecture | Paradigm | Total Params | Trainable Params | Test Accuracy | Macro F1 | Latency (GPU) | Evaluation Strategy |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **DenseNet-121** | Dense Connection | 7.22M | 7.22M | **97.29%** | **0.9729** | — | Stratified 5-Fold CV (`92.46% ± 1.14%`) |
| 🥈 | **VGG16 (Unfreeze B4+B5)** | Sequential ConvNet | 14.72M | 12.98M | **95.12%** | **0.9502** | 75.42 ms | Full Test Set (Default Live Studio Model) |
| 🥉 | **InceptionV3** | Multi-branch Inception | 21.76M | 21.76M | **95.00%** | `—*` | — | Evaluated via `model.evaluate()` |
| **#4** | **DenseNet-169** | Dense Connection | 12.76M | 12.76M | **94.32%** | `—*` | — | Evaluated via `model.evaluate()` |
| **#5** | **MobileNetV2** | Inverted Residuals | 2.19M | 2.19M | **93.94%** | `—*` | — | Evaluated via `model.evaluate()` |
| **#6** | **ResNet50 (Fine-tuned)** | Residual Skip-connection | 23.59M | 23.59M | **93.00%** | **0.9286** | **57.05 ms** | Fastest on CUDA GPU |
| **#7** | **ConvNeXt-Tiny** | Modern Pure ConvNet | 28.02M | 28.02M | **90.83%** | **0.9075** | — | Meta AI CVPR 2022 (7x7 Depthwise) |
| **#8** | **MobileNetV3-Small** | Lightweight Inverted Residual | 941K | 941K | **68.44%** | **0.6721** | 80.78 ms | Overfitting case study (100% unfreeze) |

*\* Methodological Note: Architectures marked with `—*` were benchmarked directly using Keras's built-in `model.evaluate()` on the test split (recording Test Loss & Accuracy). Multi-class confusion matrices were not exported for these three runs; in adherence to academic data integrity, Macro F1 is deliberately left unestimated.*

---

## Freezing Depth Ablation Analysis

A core algorithmic focus of this benchmark is the trade-off between **fine-tuning depth** and **distributional degradation**:

```text
VGG16 Backbone Freezing Hierarchy:
[Block 1]  →  [Block 2]  →  [Block 3]  →  [Block 4]  →  [Block 5]  →  [Classifier Head]
 ── Frozen (Low-level features) ───       ─── Adaptable ────       ─── Task-Specific ───
```

1. **Feature Extraction (100% Frozen Backbone):** Only the Dense classifier is trained (2,052 parameters). Converges safely with ImageNet features to **90.50%**.
2. **Selective Block 5 Unfreeze:** Unfreezing the deepest semantic convolution layers (`block5_conv1..3`, 7.08M trainable parameters) adapts high-level lesion boundaries, elevating accuracy to **93.00%** (+2.50%).
3. **Selective Block 4+5 Unfreeze (Optimal Point):** Unfreezing 12.98M parameters enables structural context adaptation around parenchymal margins, reaching **95.12%** (+2.12%) and Macro F1 **0.9502**.
4. **The MobileNet Overfitting Paradox:** While MobileNetV2 achieved 93.94% by freezing Batch Normalization statistics, unfreezing 100% of MobileNetV3-Small caused catastrophic distribution drift—misclassifying 210/400 meningioma cases as glioma and collapsing test accuracy from 83.84% (baseline) to 68.44%.

---

## Quickstart

### Prerequisites
- **OS:** Windows 10/11, macOS, or Linux.
- **Python:** Version 3.10 to 3.12 installed.

### Automated Setup (Windows)

```bat
git clone https://github.com/lqkiet07/brain-tumor-web.git
cd brain-tumor-web

:: Step 1: Run automated setup (creates virtual environment, installs dependencies, verifies models)
setup.bat

:: Step 2: Launch local server and open http://127.0.0.1:8010
run_demo.bat
```

*To customize the network port:*
```bat
set PORT=8020 && run_demo.bat
```

### Manual Installation (All Platforms)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch the FastAPI server
python run_demo.py
```

Navigate to `http://127.0.0.1:8010` in your web browser.

---

## Project Structure

```text
brain-tumor-web/
├── server/
│   ├── app.py             # FastAPI application, static mounts & HTTP exception handlers
│   ├── config.py          # Model registry, class label dictionary, image dimensions
│   └── model_service.py   # Tensor pre-processing, Keras session inference, heatmap generation
├── web/                   # Zero-build static frontend (HTML5 / Vanilla CSS / ES6+)
│   ├── index.html         # Application layout, dynamic leaderboard, diagnostic studio
│   ├── css/
│   │   ├── style.css      # Design token system, WCAG AA compliance, responsive grid
│   │   └── fonts.css      # Local @font-face declarations (100% offline)
│   ├── js/
│   │   ├── app.js         # Reactive leaderboard rendering & research workspace manager
│   │   └── demo.js        # Drag-and-drop handler, API client, XAI opacity controller
│   ├── data/
│   │   └── project.json   # Machine-readable single source of truth for all 8 benchmark runs
│   └── assets/            # Self-hosted woff2 fonts, confusion matrices, and XAI figures
├── models/                # Local HDF5 (.h5) model checkpoints (excluded from git tracking)
│   └── .gitkeep           # Preserves folder topology across clones
├── tests/                 # 61 automated test specifications (Pytest)
│   ├── test_api.py        # REST API contracts (/api/health, /api/models, /api/predict)
│   ├── test_heatmap.py    # Overlay dimensions, numerical stability, jet formula checks
│   ├── test_model_loading.py # Parameter count verification and layer trainable status
│   ├── test_project_data.py  # Zero-drift validation against raw CSV experiment logs
│   └── test_frontend_contract.py # Accessibility, contrast, and offline asset validation
├── tools/
│   ├── prepare_assets.py  # Asset compilation and project.json generation pipeline
│   ├── fetch_fonts.py     # Deterministic font downloader for offline deployment
│   ├── bench_api.py       # Latency benchmark measuring inference and heatmap compute
│   └── probe_model.py     # Layer-by-layer parameter inspection utility
├── setup.bat              # One-click Windows environment installer
├── run_demo.py            # Headless entrypoint script
├── requirements.txt       # Pinned library dependencies
└── pytest.ini             # Pytest test execution configuration
```

---

## Explainable AI (XAI)

In clinical neuro-oncology, model transparency is mandatory. The live diagnostic studio computes activation intensity maps directly from the deepest convolutional layer of the active model:

```python
# Intensity normalization
activation = np.mean(conv_layer_output[0], axis=-1)
activation = np.maximum(activation, 0)
activation /= (np.max(activation) + 1e-8)

# Jet colormap blending
heatmap_rgb = apply_jet_colormap(activation)
overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_rgb, 0.4, 0)
```

- **VGG16 (`block5_conv3`):** Resolves fine-grained spatial boundaries along intracranial dural margins.
- **ResNet50 (`conv5_block3_out`):** Isolates central core necrosis within higher-grade lesions while maintaining 57.05 ms GPU throughput.

---

## Testing & Verification

Every pull request and build artifact is gated by a strict 61-test automated suite ensuring full mathematical, functional, and contractual compliance:

```bash
# Execute full test suite with cache invalidation
pytest --cache-clear -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.12.7, pytest-9.1.1, pluggy-1.7.0
rootdir: brain-tumor-web
configfile: pytest.ini
collected 61 items

tests/test_api.py .........................                              [ 15%]
tests/test_assets.py ....                                                [ 21%]
tests/test_frontend_contract.py ...........                              [ 39%]
tests/test_heatmap.py ....                                               [ 45%]
tests/test_model_loading.py ....                                         [ 52%]
tests/test_prediction.py .....                                           [ 60%]
tests/test_preprocess.py ......                                          [ 70%]
tests/test_project_data.py ...............                               [ 93%]
tests/test_registry.py ..                                                [ 96%]
tests/test_report_consistency.py ..                                      [100%]

======================= 61 passed, 1 warning in 24.85s ========================
```

---

## Offline-First Architecture

Unlike typical AI demo templates that pull Google Fonts, Lucide CDN, and Tailwind runtime scripts over the wire:
1. **Self-Hosted Typography:** 27 subsets of `Be Vietnam Pro` and `IBM Plex Mono` are bundled locally in `web/assets/fonts/` (.woff2).
2. **Zero External HTTP Requests:** Audited via `tests/test_frontend_contract.py`—strictly zero outbound queries are triggered during page load or inference.
3. **Local RAM Processing:** MRI image payloads are read directly into ephemeral memory buffers, processed, and immediately garbage-collected without persistence to disk.

---

## Citation & Academic Integrity

The dataset employed across all benchmarks is derived from the public benchmark curated by Masoud Nickparvar:

```bibtex
@dataset{nickparvar2021braintumor,
  title     = {Brain Tumor MRI Dataset},
  author    = {Nickparvar, Masoud},
  year      = {2021},
  publisher = {Kaggle},
  doi       = {10.34740/KAGGLE/DSV/2645886}
}
```

### Research Disclaimer
*This system is developed strictly for algorithmic research, architectural benchmarking, and educational demonstration. It is not a certified medical device and must not be used as a substitute for professional radiological consultation or clinical diagnosis.*

---

## License

This project is licensed under the [MIT License](LICENSE).
