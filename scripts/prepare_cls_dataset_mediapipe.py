#!/usr/bin/env python3
"""
Build an Ultralytics *classification* dataset from `data_set/` using MediaPipe face cropping.

Input:
  data_set/<id>/{live,spoof}/*.jpg

Output (Ultralytics classification):
  datasets/spoof_mp_face_cls/
    train/{real,spoof}/
    val/{real,spoof}/
    test/{real,spoof}/

Notes:
- This is the "before training" pipeline you asked for: we crop a face ROI using MediaPipe,
  then resize to imgsz and save as JPEG.
- Label mapping: source `live` -> `real`, `spoof` -> `spoof`.
"""

from __future__ import annotations

import argparse
import random
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    img_path: Path
    label: str  # "real" or "spoof"


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
            label = "spoof"
        else:
            continue
        samples.append(Sample(img_path=img_path, label=label))
    return samples


def _stratified_split(samples: list[Sample], val_ratio: float, test_ratio: float, seed: int) -> dict[str, set[int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {"real": [], "spoof": []}
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


def _save_jpeg(im: Image.Image, out_path: Path, quality: int = 95) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(out_path, format="JPEG", quality=quality, optimize=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    # Ensure repo root is on PYTHONPATH so `import api` works when running:
    #   python3 scripts/prepare_cls_dataset_mediapipe.py ...
    sys.path.insert(0, str(repo_root))
    from api.mediapipe_crop import crop_best_face_mediapipe  # noqa: WPS433

    # Reduce MediaPipe/TFLite noise and avoid GPU/EGL paths in headless environments.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data_set"), help="Input dataset root")
    ap.add_argument("--output", type=Path, default=Path("datasets/spoof_mp_face_cls"), help="Output dataset root")
    ap.add_argument("--imgsz", type=int, default=224, help="Resize to square after crop")
    ap.add_argument("--val", type=float, default=0.1, help="Validation ratio")
    ap.add_argument("--test", type=float, default=0.0, help="Test ratio (0 disables)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max_total", type=int, default=0, help="Limit total images used (0 = all). Stratified by class.")
    ap.add_argument("--balance", action="store_true", help="When used with --max_total, sample ~50/50 real vs spoof.")

    # MediaPipe crop params (match your Windows-style behavior)
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
        by_cls: dict[str, list[Sample]] = {"real": [], "spoof": []}
        for s in samples:
            if s.label in by_cls:
                by_cls[s.label].append(s)

        n_real = len(by_cls["real"])
        n_spoof = len(by_cls["spoof"])
        n_all = n_real + n_spoof
        if n_all == 0:
            raise SystemExit("No labeled samples found (real/spoof).")

        if args.balance:
            tgt_real = args.max_total // 2
            tgt_spoof = args.max_total - tgt_real
        else:
            tgt_real = int(round(args.max_total * (n_real / n_all)))
            tgt_spoof = args.max_total - tgt_real

        tgt_real = min(tgt_real, n_real)
        tgt_spoof = min(tgt_spoof, n_spoof)

        remaining = args.max_total - (tgt_real + tgt_spoof)
        if remaining > 0:
            add = min(remaining, n_real - tgt_real)
            tgt_real += add
            remaining -= add
            if remaining > 0:
                add = min(remaining, n_spoof - tgt_spoof)
                tgt_spoof += add

        rng.shuffle(by_cls["real"])
        rng.shuffle(by_cls["spoof"])
        samples = by_cls["real"][:tgt_real] + by_cls["spoof"][:tgt_spoof]
        rng.shuffle(samples)

    splits = _stratified_split(samples, val_ratio=args.val, test_ratio=args.test, seed=args.seed)

    for split in ("train", "val", "test"):
        if split == "test" and args.test <= 0:
            continue
        for label in ("real", "spoof"):
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_failed = 0
    n_no_face = 0
    total_by_class: Counter[str] = Counter()
    written_by_split_class: Counter[tuple[str, str]] = Counter()

    for i, s in enumerate(samples):
        split = "train" if i in splits["train"] else "val" if i in splits["val"] else "test"
        if split == "test" and args.test <= 0:
            continue

        total_by_class[s.label] += 1
        out_name = f"{s.img_path.parent.parent.name}_{s.img_path.parent.name}_{s.img_path.stem}.jpg"
        out_path = out_root / split / s.label / out_name

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

            crop_im = Image.fromarray(cr.crop_rgb)
            if args.imgsz and args.imgsz > 0:
                crop_im = crop_im.resize((args.imgsz, args.imgsz), resample=Image.BICUBIC)
            _save_jpeg(crop_im, out_path)

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
    print(f"imgsz: {args.imgsz}")
    print(f"val_ratio: {args.val} test_ratio: {args.test} seed: {args.seed}")
    print(f"mp_min_conf: {args.mp_min_conf} mp_pad: {args.mp_pad} mp_pad_top_mult: {args.mp_pad_top_mult}")

    print("\nCLASS BALANCE (all input samples considered for output splits)")
    for cls in ("real", "spoof"):
        print(f"{cls}: {total_by_class.get(cls, 0)}")

    print("\nSPLIT COUNTS (written images)")
    for split in ("train", "val", "test"):
        if split == "test" and args.test <= 0:
            continue
        r = written_by_split_class.get((split, "real"), 0)
        s_ = written_by_split_class.get((split, "spoof"), 0)
        print(f"{split}: real={r} spoof={s_} total={r + s_}")


if __name__ == "__main__":
    main()

