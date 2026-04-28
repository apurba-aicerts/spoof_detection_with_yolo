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

---

## Decision log (high-signal, chronological)

Add new entries at the bottom. Keep each decision short.

### 2026-04-27 — Use Ultralytics YOLO classification for binary spoof detection
- **Decision**: Treat spoof detection as **image-level binary classification** (`real` vs `spoof`) using Ultralytics YOLO classification checkpoints.
- **Why**: Fast iteration, strong baselines, simple packaging (`.pt`), easy training/inference scripts.
- **Alternatives considered**: custom PyTorch training loop; video temporal models.
- **Trade-offs**: limited to single-frame cues; may need future extension for video-based liveness.

### 2026-04-27 — Dataset builder writes full resized images (not face crops)
- **Decision**: Build the classification dataset from full images resized to `--imgsz` (default 224).
- **Why**: Face box labels can be noisy; full-frame context can reduce sensitivity to imperfect boxes.
- **Trade-offs**: could include background artifacts; future iteration may add optional face-crop mode.

### 2026-04-27 — API device policy: query param + CPU fallback + `API_FORCE_CPU`
- **Decision**: API accepts `?device=cpu|0` and falls back to CPU if GPU inference fails; `API_FORCE_CPU=1` forces CPU-only serving.
- **Why**: More reliable in heterogeneous environments; avoids outages from CUDA hiccups.

### 2026-04-27 — Developer experience: comprehensive README + safe `.gitignore`
- **Decision**: Improve onboarding docs and ignore large artifacts/images by default.
- **Why**: Prevent accidental commits of datasets, prediction images, and training runs; reduce friction for new developers.

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
- Consider optional face-crop dataset mode to compare against full-image training.

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

