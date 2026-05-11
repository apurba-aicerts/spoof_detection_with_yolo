## Spoof detection (Ultralytics classification)

Binary image classification for **liveness / spoof detection** (labels: **`real`** vs **`spoof`**) using [Ultralytics](https://docs.ultralytics.com/) YOLO *classification* models.

This repo contains an end-to-end pipeline:
- Build a clean classification dataset from an existing folder structure (`data_set/`)
- Train a YOLO classification model (`.pt` weights under `runs/`)
- Run batch inference on folders for sanity checks / auditing
- Serve predictions via a small FastAPI service (with a Windows webcam client for demo use)

---

## Problem statement

In remote proctoring / identity / access workflows, attackers can attempt to bypass camera checks using printed photos, screen replays, masks, or other presentation attacks. The goal is to classify a single RGB image as:
- **`real`**: likely a live, authentic capture
- **`spoof`**: likely a presentation attack

This repo focuses on **image-level spoof classification** (not video temporal modeling) so it can be deployed as a lightweight service and composed with upstream capture systems.

---

## Goals and objectives

- **Fast developer onboarding**: a new developer can run dataset prep → train → predict → API within minutes.
- **Reproducible pipeline**: scripts with explicit flags and deterministic splitting (seeded, stratified).
- **Deployment-ready inference**: a stable API process with lazy model loading and CPU fallback.
- **Practical debugging**: easy folder prediction and “save spoof predictions” for error analysis.

Non-goals (for now):
- Video-based liveness (blink/texture over time), multi-frame aggregation, or face tracking.
- On-device/mobile deployment packaging.
- Multi-class attack taxonomy (we keep `real` vs `spoof`).

---

## Approach and methodology (high level)

- **Dataset normalization**: convert your existing `data_set/` into the folder layout expected by Ultralytics classification: `train/val/test` × `real/spoof`.
- **Label mapping**: source folders named `live` are mapped to **`real`**; `spoof` stays **`spoof`**.
- **Splitting**: **stratified split by class** with a fixed seed for repeatability.
- **Training**: fine-tune an Ultralytics YOLO classification backbone (e.g., `yolo26m-cls.pt`).
- **Inference**:
  - CLI: run inference on a folder, print counts per class, optionally copy predicted spoof images to a review folder.
  - API: FastAPI endpoints for URL-based and file-upload predictions.

---

## High-level PRD (Product Requirements Document)

### Users
- **ML engineer / developer**: trains models, runs experiments, exports artifacts.
- **Backend engineer**: serves the model behind an HTTP API and integrates with upstream systems.
- **QA / analyst**: audits predictions and reviews false positives/negatives.

### Core use cases
- **Prepare dataset** from the provided `data_set/` structure.
- **Train** a classifier and generate `best.pt` weights.
- **Predict** on ad-hoc folders for evaluation and debugging.
- **Serve** predictions via HTTP for integration and demos (including a Windows webcam client).

### Inputs and outputs
- **Input image**: RGB image (JPG/PNG/WEBP/etc.), either uploaded or fetched from a URL.
- **Output**: JSON `{ label: "real"|"spoof", confidence: float }`.

### Functional requirements
- **Dataset prep**
  - Accept input root (default `data_set/`)
  - Produce output dataset root (default `datasets/spoof_face_cls/`)
  - Resize images to a square `--imgsz` (default 224)
  - Stratified `train/val/test` split with `--seed`
  - Optional dataset size cap `--max_total` and optional `--balance` sampling
  - Supports two face-crop strategies:
    - RetinaFace `*_BB.txt` crops (offline boxes shipped with dataset)
    - MediaPipe BlazeFace crops (run detector during dataset prep)
- **Training**
  - Train from a pretrained YOLO classification checkpoint
  - Save runs under `runs/spoof_face_cls/`
- **Serving**
  - Provide `/health`, `/predict/upload`, and `/predict/url` endpoints
  - Lazy-load model on first prediction request (faster, safer startup)
  - CPU mode support and safe CUDA-disable switch for brittle environments

### Non-functional requirements (targets; adjust to your needs)
- **Latency**: single-image inference should be “interactive” on CPU; faster on GPU.
- **Reliability**: API should continue serving even if GPU inference fails (fallback to CPU).
- **Operability**: weights path and CPU/GPU behavior configurable via environment variables.

### Success metrics (suggested)
- Offline: AUROC / accuracy on a held-out set; class-wise precision/recall for `spoof`.
- Online: p95 latency, error rate, and drift monitoring on confidence distributions.

---

## Architecture overview

### Components
- **Dataset builder**: `scripts/prepare_cls_dataset.py`
- **Detection dataset builder (experimental)**: `scripts/prepare_det_dataset_mediapipe.py`
- **Trainer**: `scripts/train_cls.py`
- **Detection trainer (experimental)**: `scripts/train_det.py`
- **Batch inference**: `scripts/predict_folder_cls.py`
- **Exporter**: `scripts/export_cls.py`
- **API server**: `scripts/run_api.py` + `api/app.py`
- **Windows demo client**: `scripts/windows_webcam_client.py`
- **Webcam detection demo (experimental)**: `scripts/predict_webcam_det.py`

### Data flow

```mermaid
flowchart LR
  A[data_set/<id>/{live,spoof}/*.jpg] --> B[scripts/prepare_cls_dataset.py]
  A2[data_set/<id>/{live,spoof}/*_BB.txt] --> B
  B --> C[datasets/spoof_face_cls/{train,val,test}/{real,spoof}]
  C --> D[scripts/train_cls.py]
  D --> E[runs/spoof_face_cls/<exp>/weights/best.pt]
  E --> F[scripts/predict_folder_cls.py]
  E --> G[api/app.py (FastAPI)]
  H[Windows webcam client] -->|/predict/upload| G
  I[URL input] -->|/predict/url| G
```

### Repo layout (practical)
- **`data_set/`**: your raw dataset (already present in this workspace)
- **`datasets/`**: prepared Ultralytics datasets (created by scripts)
- **`runs/`**: training outputs and weights (Ultralytics default)
- **`scripts/`**: pipeline entrypoints (dataset prep/train/predict/export/api client)
- **`api/`**: FastAPI app and model loading/inference logic

---

## Setup

### Prerequisites
- Python **3.10+**
- Linux/WSL2 recommended (Windows is supported for the webcam *client* script)

### Install

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you ever see `"no kernel image is available for execution on the device"`, it’s typically a **torch/torchvision CUDA mismatch**. Reinstall pins from `requirements.txt`:

```bash
pip uninstall -y torch torchvision
pip install -r requirements.txt --force-reinstall
```

---

## Run the pipeline (end-to-end)

### 1) Prepare a YOLO classification dataset

Creates an Ultralytics classification dataset with the structure:
`datasets/spoof_face_cls/{train,val,test}/{real,spoof}`

#### Option A (RetinaFace `*_BB.txt`) — recommended baseline

This uses the dataset’s companion `*_BB.txt` files (RetinaFace boxes stored in a 224×224 reference space; scaled to the real image size), then crops the face ROI and resizes to `--imgsz`.

```bash
python3 scripts/prepare_cls_dataset.py \
  --input data_set \
  --output datasets/spoof_face_cls \
  --imgsz 224 \
  --min_bb_conf 0.8 \
  --val 0.1 \
  --seed 42
```

#### Option B (MediaPipe) — crop faces before training (matches Windows testing crop)

This runs **MediaPipe BlazeFace** on each full image, crops the best face, expands the box using padding (to include more head), then resizes to `--imgsz`.

```bash
python3 scripts/prepare_cls_dataset_mediapipe.py \
  --input data_set \
  --output datasets/spoof_mp_face_cls \
  --imgsz 224 \
  --val 0.1 \
  --seed 42 \
  --mp_min_conf 0.6 \
  --mp_pad 0.20 \
  --mp_pad_top_mult 1.5
```

Optional controls:
- **Limit total images** (stratified by class):

```bash
python3 scripts/prepare_cls_dataset.py --max_total 10000
```

- **Force a balanced subset** (~50/50):

```bash
python3 scripts/prepare_cls_dataset.py --max_total 10000 --balance
```

- **Create a test split** (default is disabled):

```bash
python3 scripts/prepare_cls_dataset.py --test 0.1
```

These flags also work with the MediaPipe dataset builder:

```bash
python3 scripts/prepare_cls_dataset_mediapipe.py --max_total 10000 --balance --test 0.1
```

### Dataset sanity checks (recommended)

After building a dataset, it’s worth doing a quick scan to confirm:
- expected folder structure exists (`train|val|test` × `real|spoof`)
- counts look reasonable per split/class
- no unexpected non-image files are mixed into class folders

Example (one-pass recursive scan for `datasets/spoof_mp_face_cls_30k/`):

```bash
python3 - <<'PY'
from pathlib import Path
from collections import Counter

root = Path("datasets/spoof_mp_face_cls_30k")
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

counts = Counter()
non_images = []

for p in root.rglob("*"):
    if not p.is_file():
        continue
    if ":" in p.name:  # Windows ADS artifacts like :Zone.Identifier (rare on WSL)
        continue
    if p.suffix.lower() in img_exts:
        rel = p.relative_to(root).parts
        split = rel[0] if len(rel) > 0 else "UNKNOWN"
        label = rel[1] if len(rel) > 1 else "UNKNOWN"
        counts[(split, label)] += 1
    else:
        non_images.append(str(p))

print("root:", root.resolve())
print("total_images:", sum(counts.values()))
for split in ("train", "val", "test"):
    for label in ("real", "spoof"):
        print(f"{split}/{label}:", counts.get((split, label), 0))
if non_images:
    print("\\nnon-image files (examples):")
    for x in non_images[:10]:
        print(" -", x)
PY
```

Note on `*.cache`: Ultralytics may create dataset cache files like `train.cache` / `val.cache` at the dataset root. These are **not images** and are safe to delete; Ultralytics will regenerate them.

### 2) Train

Trains a YOLO classification model and writes outputs under `runs/spoof_face_cls/` (or whichever dataset you built).

```bash
python3 scripts/train_cls.py \
  --data datasets/spoof_face_cls \
  --model yolo26m-cls.pt \
  --epochs 50 \
  --imgsz 224 \
  --device auto
```

Notes:
- On WSL2, dataloader workers default to **0** for stability (see `--workers` flag).
- `--device auto` uses GPU if available, else CPU.

### 3) Predict on a folder (counts + optional “save spoofs”)

```bash
python3 scripts/predict_folder_cls.py \
  --weights runs/spoof_face_cls/exp-face-only/weights/best.pt \
  --source test_image
```

To also copy all images predicted as spoof into a review folder:

```bash
python3 scripts/predict_folder_cls.py \
  --weights runs/spoof_face_cls/exp-face-only/weights/best.pt \
  --source test_image \
  --spoof_dir runs/spoof_face_cls/spoof_test
```

### 4) Export (ONNX example)

```bash
python3 scripts/export_cls.py \
  --weights runs/spoof_face_cls/exp-face-only/weights/best.pt \
  --format onnx \
  --imgsz 224 \
  --device cpu
```

---

## Experimental: single-stage YOLO detection (real vs fake faces)

This experiment replaces the 2-stage pipeline:

```text
Face Detection → Face Crop → Classification
```

with a **single YOLO detection model** that predicts:
- face bounding boxes
- class: `real` vs `fake`

### 1) Prepare a YOLO detection dataset (MediaPipe auto-annotations)

This generates YOLO-format labels from MediaPipe face detections:

- **Dataset root**: `datasets/spoof_face_det_mp/`
- **Structure**:
  - `images/{train,val,test}/...`
  - `labels/{train,val,test}/...`
  - `dataset.yaml`
- **Classes**: `0=real` (source folder `live`), `1=fake` (source folder `spoof`)
- **Label format**: `class_id x_center y_center width height` (all normalized 0–1)

```bash
python3 scripts/prepare_det_dataset_mediapipe.py \
  --input data_set \
  --output datasets/spoof_face_det_mp \
  --val 0.1 \
  --test 0.1 \
  --seed 42 \
  --mp_min_conf 0.6 \
  --mp_pad 0.20 \
  --mp_pad_top_mult 1.5
```

Notes:
- The script writes **one box per image** (the “best” detected face).
- If MediaPipe can’t find a face, the image is counted as `no_face` and skipped.

### 2) Train a YOLO detection model

```bash
python3 scripts/train_det.py \
  --data datasets/spoof_face_det_mp/dataset.yaml \
  --model yolo26n.pt \
  --epochs 100 \
  --imgsz 640 \
  --device auto
```

### 3) Real-time webcam inference (local overlay)

Install webcam deps (Linux/WSL GUI required):

```bash
pip install opencv-python
```

Run:

```bash
python3 scripts/predict_webcam_det.py \
  --weights runs/spoof_face_det/exp/weights/best.pt \
  --cam 0 \
  --imgsz 640 \
  --conf 0.25
```

---

## Serving (FastAPI)

### Start the API server

```bash
python3 scripts/run_api.py
```

Defaults (from `scripts/run_api.py`):
- **HOST**: `0.0.0.0`
- **PORT**: `8011`

The FastAPI app itself loads weights from `SPOOF_WEIGHTS` if set, otherwise defaults to:
- `runs/spoof_face_cls/exp-face-only/weights/best.pt` (see `api/app.py`)

Override weights:

```bash
SPOOF_WEIGHTS="runs/spoof_face_cls/exp-face-only/weights/best.pt" python3 scripts/run_api.py
```

### Endpoints

- **Health**

```bash
curl "http://127.0.0.1:8011/health"
```

- **Predict from an image URL**

```bash
curl -X POST "http://127.0.0.1:8011/predict/url?device=cpu" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}'
```

- **Predict from an uploaded file**

```bash
curl -X POST "http://127.0.0.1:8011/predict/upload?device=cpu" \
  -F "file=@/path/to/image.jpg"
```

### CPU/GPU behavior (important)

- The API supports `device=cpu` or `device=0` (GPU 0) via query param.
- If GPU inference fails for a request, the API will **fallback to CPU**.
- If you want to force the API process into CPU-only mode (recommended for brittle CUDA setups):

```bash
API_FORCE_CPU=1 python3 scripts/run_api.py
```

---

## Windows webcam demo client

Run `scripts/windows_webcam_client.py` on **Windows** (not inside WSL).

Install dependencies:

```bash
pip install opencv-python requests
```

Webcam mode (streams frames to the API):

```bash
python scripts/windows_webcam_client.py --api http://127.0.0.1:8011
```

URL mode (single-shot or interactive):

```bash
python scripts/windows_webcam_client.py --api http://127.0.0.1:8011 --mode url --url "https://example.com/image.jpg"
```

---

## Troubleshooting

- **Weights not found**
  - Ensure `SPOOF_WEIGHTS` points to an existing `best.pt`, or train a model first.
  - Check `/health` to see what weights path the API expects.
- **`train.cache` / `val.cache` appears in your dataset folder**
  - These are Ultralytics dataset cache files, not corruption. Safe to delete; they will be regenerated.
- **WSL stability during training**
  - Use fewer dataloader workers (`--workers 0` is the default on WSL).
- **CUDA issues in the API**
  - Set `API_FORCE_CPU=1` to prevent CUDA probing and force CPU inference.
- **Dataset looks wrong / label mapping**
  - The dataset builder maps source folder `live` → label `real`, and `spoof` → label `spoof`.


