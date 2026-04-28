# Development log & progress tracker

This document is a **living development footprint** for this spoof-detection project. It is meant to answer:
- **What changed, when, and why?**
- **What challenges showed up and how were they addressed?**
- **What decisions were made and what alternatives were considered?**
- **What did we learn? What’s next?**

It is also structured so you can quickly produce **weekly status reports** by copying the “Weekly report snippet” section for a given week.

---

## How to use this log (workflow)

- **Append, don’t rewrite**: add a new entry when you make a meaningful change (new script, API behavior change, model/data change, export/deploy change).
- **Prefer “why” over “what”**: link changes to goals, constraints, and observed failures.
- **Keep entries small and timestamped**: 5–15 lines is usually enough.
- **Tag decisions** with `Decision:` so they’re easy to scan later.

---

## Project initiation (origin story)

### Context
We needed a lightweight, deployable **real vs spoof** classifier that can be trained from an existing internal dataset layout and served behind an HTTP API for integration and demo workflows (including Windows webcam streaming into a WSL-hosted API).

### Starting point (repo capabilities at baseline)
At the time this log was created, the repo already contained:
- A dataset builder converting `data_set/<id>/{live,spoof}` into Ultralytics classification format `datasets/spoof_cls/{train,val,test}/{real,spoof}`.
- Training, export, and folder prediction scripts under `scripts/`.
- A FastAPI app under `api/` served via Hypercorn (`scripts/run_api.py`).
- A Windows webcam client to stream frames to the API (`scripts/windows_webcam_client.py`).

---

## Project timeline (quick index)

Use this section to quickly understand the order of major changes. Details live in the decision log entries below.

- **2026-04-27**: Established Ultralytics YOLO *classification* baseline (binary `real` vs `spoof`) + dataset builder + API serving (full-image resized baseline).
- **2026-04-28**: Switched to **face ROI** training; introduced two reproducible dataset-prep strategies:
  - **RetinaFace `*_BB.txt`** face crops (offline boxes in dataset, scaled from 224-ref space)
  - **MediaPipe BlazeFace** face crops (run detector during dataset prep to match testing crop behavior)

---

## Key challenges & lessons learned (running list)

Keep this section updated as you encounter new classes of problems.

### CUDA / Torch stability in mixed environments
- **Problem**: CUDA driver/stub mismatches can surface as errors like `"no kernel image is available for execution on the device"` and can crash or hang initialization.
- **Strategy**:
  - Pin `torch`/`torchvision` versions in `requirements.txt`.
  - Add an explicit **CPU safety switch** (`API_FORCE_CPU=1`) for serving environments.
  - Lazy-load the model on first request to reduce startup blast radius.
- **Lesson**: Production inference needs a predictable device strategy; “auto” is useful for training, but serving often benefits from explicit CPU/GPU policy + fallback.

### WSL2 dataloader instability (/dev/shm constraints)
- **Problem**: Dataloader workers and shared memory constraints can cause intermittent training failures/hangs on WSL2.
- **Strategy**: Default dataloader workers to **0 on WSL** (configurable via `--workers`) and keep training stable by default.
- **Lesson**: Optimize for “works everywhere” defaults; allow power users to tune up workers/batch later.

### API server crashes with some ASGI stacks
- **Problem**: Rare native crashes can occur depending on the server stack and CUDA probing/initialization.
- **Decision**: Serve with **Hypercorn** in `scripts/run_api.py`, and keep the FastAPI app lightweight at import time (lazy model load).
- **Lesson**: In ML services, import-time side effects (CUDA probing, model init) are a common reliability risk.

### Preprocessing mismatch can invalidate “high accuracy”
- **Problem**: Confusion-matrix accuracy can look excellent on a held-out split, but **unknown full-frame images** can be predicted as `spoof` very frequently.
- **Root cause**: Training and inference used **different preprocessing** (face ROI vs full-frame), causing distribution shift.
- **Lesson**: For trustworthy evaluation, ensure **training preprocessing == inference preprocessing** (same face detection/crop strategy and similar input domain).

### MediaPipe dataset prep can be resource-heavy
- **Problem**: Large dataset prep runs with MediaPipe can get killed (exit code 137) if the face detector is repeatedly initialized.
- **Strategy**:
  - Cache/reuse MediaPipe detector instances during dataset prep (avoid per-image initialization).
  - Start with smaller `--max_total` to validate crop quality and stability, then scale up.

---

## Decision log (high-signal, chronological)

Add new entries at the bottom. Keep each decision short.

### 2026-04-27 — Use Ultralytics YOLO classification for binary spoof detection
- **Decision**: Treat spoof detection as **image-level binary classification** (`real` vs `spoof`) using Ultralytics YOLO classification checkpoints.
- **Why**: Fast iteration, strong baselines, simple packaging (`.pt`), easy training/inference scripts.
- **Alternatives considered**: custom PyTorch training loop; video temporal models.
- **Trade-offs**: limited to single-frame cues; may need future extension for video-based liveness.
- **Evidence**:
  - Code: `scripts/train_cls.py`, `scripts/predict_folder_cls.py`, `scripts/export_cls.py`
  - Dependencies: `requirements.txt` (includes `ultralytics`)

### 2026-04-27 — Dataset builder writes full resized images (not face crops)
- **Decision**: Build the classification dataset from full images resized to `--imgsz` (default 224).
- **Why**: Face box labels can be noisy; full-frame context can reduce sensitivity to imperfect boxes.
- **Trade-offs**: could include background artifacts; future iteration may add optional face-crop mode.
- **Evidence**:
  - Code: `scripts/prepare_cls_dataset.py` (resize + write JPEG)
  - Docs: `README.md` (dataset prep section describing resizing)

### 2026-04-27 — API device policy: query param + CPU fallback + `API_FORCE_CPU`
- **Decision**: API accepts `?device=cpu|0` and falls back to CPU if GPU inference fails; `API_FORCE_CPU=1` forces CPU-only serving.
- **Why**: More reliable in heterogeneous environments; avoids outages from CUDA hiccups.
- **Evidence**:
  - Code: `api/app.py` (`_resolve_device()`, GPU→CPU fallback)
  - Runner: `scripts/run_api.py` (`API_FORCE_CPU`, Hypercorn)

### 2026-04-27 — Developer experience: comprehensive README + safe `.gitignore`
- **Decision**: Improve onboarding docs and ignore large artifacts/images by default.
- **Why**: Prevent accidental commits of datasets, prediction images, and training runs; reduce friction for new developers.
- **Evidence**:
  - Docs: `README.md`
  - Repo hygiene: `.gitignore`

---

## Weekly progress tracker

Add one section per week. Keep it honest, measurable, and scoped.

### Week of 2026-04-27

#### Summary (1–2 sentences)
End-to-end pipeline is functional: dataset prep → training → inference → API serving; developer docs and repo hygiene improved for onboarding.

#### Shipped / completed
- Dataset preparation script supports stratified splits, size limiting, and optional balancing.
- Training script supports auto device selection and WSL-stable defaults.
- Folder prediction script reports counts and can copy predicted spoof images for review.
- FastAPI service supports `/health`, `/predict/url`, `/predict/upload` with lazy model loading.
- Windows webcam client supports streaming frames to the API.
- README updated to include problem statement, PRD, architecture, and run instructions.
- `.gitignore` added to prevent committing datasets/artifacts/images.

#### Key challenges encountered
- Serving stability in environments with brittle CUDA setups.
- WSL2 training stability with dataloader worker defaults.

#### Strategies & fixes applied
- Introduced `API_FORCE_CPU` and CPU fallback behavior in API inference path.
- Defaulted training workers to 0 on WSL2.
- Served API via Hypercorn to reduce native-crash surface area.

#### Decisions made (links to decision log)
- Ultralytics classification baseline (binary `real` vs `spoof`).
- Full-image dataset generation at `imgsz=224`.
- API device policy: explicit + fallback.

#### Metrics / evidence (if available)
- Add: training metrics snapshot, confusion matrix, or validation accuracy for the current best run.
- Add: p95 latency for `/predict/upload` on CPU/GPU.

#### Next (1–2 weeks)
- Add an evaluation script (confusion matrix, ROC/AUC, per-class precision/recall).
- Add a small “golden” test set and a regression check for the API output schema.
- Add model/version metadata to API responses (weights hash/run id) for traceability.
- Add a second dataset-prep option using MediaPipe (inference-style crop) and compare against RetinaFace `*_BB.txt` crops.

#### Ongoing tasks (carryover)
- Document dataset provenance and labeling guidelines (what counts as spoof).
- Decide deployment target (Docker, systemd, k8s) and add a minimal deployment recipe.

#### Weekly report snippet (copy/paste)
**Week of 2026-04-27**
- **Shipped**: End-to-end spoof classification pipeline + API + Windows webcam client. Improved onboarding README and added `.gitignore` to prevent committing datasets/images/runs.
- **Challenges**: CUDA/torch mismatches and WSL2 dataloader stability.
- **Mitigations**: CPU safety switch (`API_FORCE_CPU`), CPU fallback on GPU failure, WSL-stable worker defaults, Hypercorn serving.
- **Next**: add evaluation tooling + golden test set + API metadata for traceability.

---

## Entry template (use for ad-hoc updates)

Copy/paste this when you make a meaningful change mid-week.

### YYYY-MM-DD — <short title>
- **What changed**:
- **Why**:
- **Impact**:
- **Risks / trade-offs**:
- **Follow-ups**:

### 2026-04-28 — Switch to face-only training + add MediaPipe-before-training option
- **Decision**: Standardize the project around **face ROI** inputs (not full-frame) to reduce background leakage and align training/testing preprocessing.
- **What changed**:
  - `scripts/prepare_cls_dataset.py` now builds `datasets/spoof_face_cls` by **cropping with RetinaFace `*_BB.txt`** (224-ref coords scaled to image size).
  - Added `scripts/prepare_cls_dataset_mediapipe.py` to build `datasets/spoof_mp_face_cls` by **running MediaPipe BlazeFace during dataset prep** (pad + extra top padding to include head, matching testing behavior).
  - Updated training defaults to `datasets/spoof_face_cls` and outputs under `runs/spoof_face_cls`.
  - Updated API default weights to a face-only run and added `expects_face_crop` to `/health`.
- **Why**: Confusion-matrix accuracy only holds when test images match training preprocessing. Full-frame “unknown images” were producing unreliable results due to mismatch.
- **Impact**: Clearer single “happy path” for developers, plus an experimental MediaPipe dataset builder to match real testing conditions.
- **Risks / trade-offs**: MediaPipe dataset prep is slower and can be resource-heavy; keep `--max_total` while iterating and scale up after validating crop quality.
- **Evidence**:
  - RetinaFace crop builder: `scripts/prepare_cls_dataset.py` (reads `*_BB.txt`, applies 224-ref scaling)
  - MediaPipe crop builder: `scripts/prepare_cls_dataset_mediapipe.py`
  - MediaPipe cropper: `api/mediapipe_crop.py`
  - Training defaults: `scripts/train_cls.py` (defaults point to `datasets/spoof_face_cls`)
  - API default weights + health schema: `api/app.py` (`DEFAULT_WEIGHTS`, `/health`)
  - Docs: `README.md` (Option A/Option B dataset prep)

### 2026-04-28 — Decision detail: Why MediaPipe for face detection/cropping (instead of dataset-provided face info)
- **Decision**: Support **MediaPipe BlazeFace** as a first-class *before-training* face crop strategy (in addition to dataset `*_BB.txt` crops).
- **Problem observed**: High validation accuracy did not translate to reliable behavior on arbitrary “unknown” images when the input preprocessing differed from training (e.g., full-frame inputs).
- **Why MediaPipe helps**:
  - **Training/testing alignment**: We introduced a MediaPipe-based cropper for testing (Windows client style). Training on MediaPipe crops reduces preprocessing mismatch when that same cropper is used.
  - **Independence from dataset annotations**: Dataset `*_BB.txt` boxes might be missing/low-confidence for some images; MediaPipe can still produce a usable ROI when a face is visible.
  - **Operational simplicity**: MediaPipe is CPU-friendly and easy to run anywhere (including Windows clients), which makes it useful for consistent ROI extraction across environments.
- **Why not rely only on dataset `*_BB.txt`**:
  - **Mismatch risk**: `*_BB.txt` boxes are produced by a different detector (RetinaFace) and stored in a 224-ref coordinate space; mistakes in scaling or detector differences can drift crop distributions from what we test with.
  - **Portability**: At inference time for arbitrary images, we don’t have `*_BB.txt` files; we must detect faces anyway.
- **Trade-offs / risks**:
  - **Speed**: MediaPipe dataset prep is slower than using precomputed boxes.
  - **Resource behavior**: Detector initialization can be expensive; we must reuse detector instances and scale up gradually.
  - **Distribution changes**: Switching detectors changes the crop distribution; results must be re-evaluated (do not compare metrics across pipelines without noting crop strategy).
- **Reproducible commands**:
  - RetinaFace-box crops:
    - `python3 scripts/prepare_cls_dataset.py --input data_set --output datasets/spoof_face_cls --imgsz 224 --min_bb_conf 0.8 --val 0.1 --test 0.1 --seed 42`
  - MediaPipe crops:
    - `./spoof_env/bin/python scripts/prepare_cls_dataset_mediapipe.py --input data_set --output datasets/spoof_mp_face_cls --imgsz 224 --mp_min_conf 0.6 --mp_pad 0.20 --mp_pad_top_mult 1.5 --val 0.1 --test 0.1 --seed 42 --max_total 10000 --balance`
- **Evidence**:
  - MediaPipe dataset builder + args: `scripts/prepare_cls_dataset_mediapipe.py`
  - MediaPipe crop algorithm (pad/top padding): `api/mediapipe_crop.py`
  - Evidence of dataset `*_BB.txt` assumption + scaling: `data_set/README` and `scripts/prepare_cls_dataset.py`


