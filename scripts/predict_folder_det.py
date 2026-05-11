#!/usr/bin/env python3
"""Run batch predictions with a YOLO *detection* model and optionally save predicted fake images.

This mirrors `scripts/predict_folder_cls.py`, but for detection models trained to output
face boxes with classes: 0=real, 1=fake.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

# Avoid rare native crashes on some Linux/WSL builds (OpenMP/BLAS thread teardown).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from ultralytics import YOLO


def _summarize_detection_result(r, *, fake_name: str = "fake") -> tuple[str, float]:
    """
    Convert a detection Results object into a single (label, confidence) summary.

    Policy:
    - If there are any detections of class `fake_name`, return the highest-confidence fake.
    - Else return the highest-confidence detection of any class.
    - If there are no detections, return ("no_face", 0.0).
    """
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ("no_face", 0.0)

    names = getattr(r, "names", {}) or {}
    best_any = ("unknown", 0.0)
    best_fake = ("fake", 0.0)

    for b in boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        name = str(names.get(cls_id, cls_id))
        if conf > best_any[1]:
            best_any = (name, conf)
        if str(name).lower() == fake_name.lower() and conf > best_fake[1]:
            best_fake = (name, conf)

    if best_fake[1] > 0.0:
        return best_fake
    return best_any


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    ap.add_argument("--source", type=Path, default=Path("test_image"), help="Folder of images")
    ap.add_argument(
        "--fake_dir",
        type=Path,
        default=Path("test_result"),
        help="Where to save images predicted as fake (relative to repo by default)",
    )
    ap.add_argument("--fake_name", type=str, default="fake", help="Class name to treat as fake/spoof")
    ap.add_argument("--conf", type=float, default=0.8, help="Detection confidence threshold")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | 0 | auto")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    weights = (repo_root / args.weights).resolve() if not args.weights.is_absolute() else args.weights.resolve()
    source = (repo_root / args.source).resolve() if not args.source.is_absolute() else args.source.resolve()
    fake_dir = (repo_root / args.fake_dir).resolve() if not args.fake_dir.is_absolute() else args.fake_dir.resolve()

    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    try:
        import torch  # type: ignore

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        verbose=False,
        conf=float(args.conf),
        # device=str(args.device),
        save=True,
        stream=True,
    )

    counts: Counter[str] = Counter()
    total = 0
    saved_fake = 0

    fake_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        label, conf = _summarize_detection_result(r, fake_name=args.fake_name)
        counts[label] += 1
        total += 1

        if str(label).lower() == args.fake_name.lower():
            src_path = Path(getattr(r, "path", ""))
            if src_path.exists():
                dst = fake_dir / src_path.name
                if dst.exists():
                    dst = fake_dir / f"{src_path.stem}__{saved_fake}{src_path.suffix}"
                shutil.copy2(src_path, dst)
                saved_fake += 1

    print(f"weights: {weights}")
    print(f"source: {source}")
    print(f"conf: {args.conf} device: {args.device}")
    print(f"total_images_predicted: {total}")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k}: {v}")
    print(f"fake_saved_dir: {fake_dir}")
    print(f"fake_saved_count: {saved_fake}")


if __name__ == "__main__":
    main()

