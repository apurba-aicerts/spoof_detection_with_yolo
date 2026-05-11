#!/usr/bin/env python3
"""
Build an Ultralytics *detection* dataset for spoof detection using MediaPipe face detection.

We treat spoof detection as 2-class object detection:
  - class 0: real  (source folder name contains "live")
  - class 1: fake  (source folder name contains "spoof")

Input:
  data_set/<id>/{live,spoof}/*.(jpg|png|...)

Output:
  datasets/spoof_face_det_mp/
    images/{train,val,test}/...
    labels/{train,val,test}/...   # YOLO txt files: class x_center y_center w h (normalized)
    dataset.yaml

Notes:
- Uses the *best* detected face per image (highest score, tie-broken by area).
- Bounding box uses the padded MediaPipe box (pad + extra top pad) to include more head context.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAME_TO_ID = {"real": 0, "fake": 1}


@dataclass(frozen=True)
class Sample:
    img_path: Path
    label: str  # "real" or "fake"


def _is_ads_artifact(p: Path) -> bool:
    return ":" in p.name


def _iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _is_ads_artifact(p):
            continue
        if p.suffix.lower() in IMG_EXTS:
            yield p


def _collect_samples(dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for img_path in _iter_images(dataset_root):
        parts = {p.lower() for p in img_path.parts}
        if "live" in parts:
            label = "real"
        elif "spoof" in parts:
            label = "fake"
        else:
            continue
        samples.append(Sample(img_path=img_path, label=label))
    return samples


def _stratified_split(samples: list[Sample], val_ratio: float, test_ratio: float, seed: int) -> dict[str, set[int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {"real": [], "fake": []}
    for i, s in enumerate(samples):
        if s.label in by_label:
            by_label[s.label].append(i)

    splits: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    for indices in by_label.values():
        rng.shuffle(indices)
        n = len(indices)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        splits["test"] |= set(indices[:n_test])
        splits["val"] |= set(indices[n_test : n_test + n_val])
        splits["train"] |= set(indices[n_test + n_val :])
    return splits


def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, *, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    if img_w <= 0 or img_h <= 0:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    xc = ((x1 + x2) / 2.0) / float(img_w)
    yc = ((y1 + y2) / 2.0) / float(img_h)
    w = (x2 - x1) / float(img_w)
    h = (y2 - y1) / float(img_h)
    # clamp for safety
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return (xc, yc, w, h)


def _write_yaml(out_root: Path) -> None:
    # Ultralytics supports relative paths inside the dataset root.
    yml = "\n".join(
        [
            f"path: {out_root.resolve()}",
            "",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: real",
            "  1: fake",
            "",
        ]
    )
    (out_root / "dataset.yaml").write_text(yml, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from api.mediapipe_crop import crop_best_face_mediapipe  # noqa: WPS433

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data_set"), help="Input dataset root")
    ap.add_argument("--output", type=Path, default=Path("datasets/spoof_face_det_mp"), help="Output dataset root")
    ap.add_argument("--val", type=float, default=0.1, help="Validation ratio")
    ap.add_argument("--test", type=float, default=0.1, help="Test ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max_total", type=int, default=0, help="Limit total images used (0 = all). Stratified by class.")
    ap.add_argument("--balance", action="store_true", help="When used with --max_total, sample ~50/50 real vs fake.")

    ap.add_argument("--mp_min_conf", type=float, default=0.6, help="MediaPipe min detection confidence")
    ap.add_argument("--mp_pad", type=float, default=0.20, help="Pad ratio around face box")
    ap.add_argument("--mp_pad_top_mult", type=float, default=1.5, help="Extra top padding multiplier")
    args = ap.parse_args()

    in_root = (repo_root / args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    out_root = (repo_root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if not in_root.exists():
        raise SystemExit(f"Input not found: {in_root}")

    samples = _collect_samples(in_root)
    if not samples:
        raise SystemExit(f"No images found under: {in_root}")

    # Optional sampling
    if args.max_total and args.max_total > 0 and args.max_total < len(samples):
        rng = random.Random(args.seed)
        by_cls: dict[str, list[Sample]] = {"real": [], "fake": []}
        for s in samples:
            if s.label in by_cls:
                by_cls[s.label].append(s)

        n_real = len(by_cls["real"])
        n_fake = len(by_cls["fake"])
        n_all = n_real + n_fake
        if n_all == 0:
            raise SystemExit("No labeled samples found (real/fake).")

        if args.balance:
            tgt_real = args.max_total // 2
            tgt_fake = args.max_total - tgt_real
        else:
            tgt_real = int(round(args.max_total * (n_real / n_all)))
            tgt_fake = args.max_total - tgt_real

        tgt_real = min(tgt_real, n_real)
        tgt_fake = min(tgt_fake, n_fake)
        remaining = args.max_total - (tgt_real + tgt_fake)
        if remaining > 0:
            add = min(remaining, n_real - tgt_real)
            tgt_real += add
            remaining -= add
            if remaining > 0:
                add = min(remaining, n_fake - tgt_fake)
                tgt_fake += add

        rng.shuffle(by_cls["real"])
        rng.shuffle(by_cls["fake"])
        samples = by_cls["real"][:tgt_real] + by_cls["fake"][:tgt_fake]
        rng.shuffle(samples)

    splits = _stratified_split(samples, val_ratio=float(args.val), test_ratio=float(args.test), seed=int(args.seed))

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    _write_yaml(out_root)

    n_written = 0
    n_failed = 0
    n_no_face = 0
    written_by_split_class: Counter[tuple[str, str]] = Counter()

    for i, s in enumerate(samples):
        split = "train" if i in splits["train"] else "val" if i in splits["val"] else "test"
        try:
            im = Image.open(s.img_path)
            im.load()
            im = im.convert("RGB")
            rgb = np.array(im, dtype=np.uint8)
            cr = crop_best_face_mediapipe(
                rgb,
                min_conf=float(args.mp_min_conf),
                model_selection=1,
                pad=float(args.mp_pad),
                pad_top_mult=float(args.mp_pad_top_mult),
            )
            if cr is None:
                n_no_face += 1
                raise ValueError("no_face")

            h, w = rgb.shape[:2]
            x1, y1, x2, y2 = cr.xyxy_padded
            yolo = _xyxy_to_yolo(x1, y1, x2, y2, img_w=w, img_h=h)
            if yolo is None:
                n_failed += 1
                continue

            class_id = CLASS_NAME_TO_ID[s.label]

            # Keep original file bytes (avoid recompressing) by copying.
            out_stem = f"{s.img_path.parent.parent.name}_{s.img_path.parent.name}_{s.img_path.stem}"
            out_img = out_root / "images" / split / f"{out_stem}{s.img_path.suffix.lower()}"
            out_lbl = out_root / "labels" / split / f"{out_stem}.txt"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)

            out_img.write_bytes(s.img_path.read_bytes())
            xc, yc, bw, bh = yolo
            out_lbl.write_text(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

            n_written += 1
            written_by_split_class[(split, s.label)] += 1
        except Exception:
            n_failed += 1

    print("DONE")
    print(f"input_root: {in_root}")
    print(f"output_root: {out_root}")
    print(f"samples_total: {len(samples)}")
    print(f"written: {n_written}")
    print(f"failed: {n_failed}")
    print(f"no_face: {n_no_face}")
    print(f"max_total: {args.max_total}")
    print(f"balance: {bool(args.balance)}")
    print(f"val_ratio: {args.val} test_ratio: {args.test} seed: {args.seed}")
    print(f"mp_min_conf: {args.mp_min_conf} mp_pad: {args.mp_pad} mp_pad_top_mult: {args.mp_pad_top_mult}")

    print("\nSPLIT COUNTS (written images)")
    for split in ("train", "val", "test"):
        r = written_by_split_class.get((split, "real"), 0)
        f = written_by_split_class.get((split, "fake"), 0)
        print(f"{split}: real={r} fake={f} total={r + f}")


if __name__ == "__main__":
    main()

