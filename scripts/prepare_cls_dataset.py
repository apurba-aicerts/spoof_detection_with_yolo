#!/usr/bin/env python3
"""
Build an Ultralytics *classification* dataset from the existing `data_set/` layout.

Input (current workspace):
  data_set/<split_id>/{live,spoof}/*.jpg
  data_set/<split_id>/{live,spoof}/*_BB.txt   (bounding box: x y w h conf)

Output (Ultralytics classification):
  datasets/spoof_cls/
    train/{real,spoof}/
    val/{real,spoof}/
    test/{real,spoof}/

Notes:
- We ignore Windows ADS artifacts like `:Zone.Identifier`.
- This script builds **full-image** classification splits (real vs spoof).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from collections import Counter


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    img_path: Path
    label: str  # "real" or "spoof"


def _is_ads_artifact(p: Path) -> bool:
    # WSL sometimes shows Windows ADS artifacts like `file.jpg:Zone.Identifier`
    return ":" in p.name


def _iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _is_ads_artifact(p):
            continue
        if p.suffix.lower() in IMG_EXTS:
            yield p


def _save_jpeg(im: Image.Image, out_path: Path, quality: int = 95) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(out_path, format="JPEG", quality=quality, optimize=True)


def _collect_samples(dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for img_path in _iter_images(dataset_root):
        # Expect structure: .../live/*.jpg or .../spoof/*.jpg
        parts = {p.lower() for p in img_path.parts}
        if "live" in parts:
            label = "real"
        elif "spoof" in parts:
            label = "spoof"
        else:
            continue
        samples.append(Sample(img_path=img_path, label=label))
    return samples


def _split_indices(n: int, val_ratio: float, test_ratio: float, rng: random.Random) -> tuple[set[int], set[int], set[int]]:
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = set(idx[:n_test])
    val = set(idx[n_test : n_test + n_val])
    train = set(idx[n_test + n_val :])
    return train, val, test


def _stratified_split(samples: list[Sample], val_ratio: float, test_ratio: float, seed: int) -> dict[str, set[int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {"real": [], "spoof": []}
    for i, s in enumerate(samples):
        if s.label in by_label:
            by_label[s.label].append(i)

    splits: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    for label, indices in by_label.items():
        rng.shuffle(indices)
        n = len(indices)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        test = set(indices[:n_test])
        val = set(indices[n_test : n_test + n_val])
        train = set(indices[n_test + n_val :])

        splits["train"] |= train
        splits["val"] |= val
        splits["test"] |= test
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data_set"), help="Input dataset root")
    ap.add_argument("--output", type=Path, default=Path("datasets/spoof_cls"), help="Output dataset root")
    ap.add_argument("--imgsz", type=int, default=224, help="Resize to square")
    ap.add_argument("--val", type=float, default=0.1, help="Validation ratio")
    ap.add_argument("--test", type=float, default=0.0, help="Test ratio (0 disables)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--max_total",
        type=int,
        default=0,
        help="Limit total images used (0 = use all). Sampling is stratified by class.",
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="When used with --max_total, sample ~50/50 real vs spoof (as much as available).",
    )
    args = ap.parse_args()

    in_root = args.input.resolve()
    out_root = args.output.resolve()

    if not in_root.exists():
        raise SystemExit(f"Input not found: {in_root}")

    samples = _collect_samples(in_root)
    if not samples:
        raise SystemExit(f"No images found under: {in_root}")

    # Optional: limit dataset size (either proportional or balanced).
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
            # Balanced targets (50/50)
            tgt_real = args.max_total // 2
            tgt_spoof = args.max_total - tgt_real
        else:
            # Proportional targets (rounded) + remainder distribution.
            tgt_real = int(round(args.max_total * (n_real / n_all)))
            tgt_spoof = args.max_total - tgt_real

        tgt_real = min(tgt_real, n_real)
        tgt_spoof = min(tgt_spoof, n_spoof)

        # If one class is capped, give remaining budget to the other class.
        remaining = args.max_total - (tgt_real + tgt_spoof)
        if remaining > 0:
            # In balance mode, fill from whichever class still has capacity.
            # In proportional mode, also fill from whichever has capacity.
            add = min(remaining, n_real - tgt_real)
            tgt_real += add
            remaining -= add
            if remaining > 0:
                add = min(remaining, n_spoof - tgt_spoof)
                tgt_spoof += add
                remaining -= add

        rng.shuffle(by_cls["real"])
        rng.shuffle(by_cls["spoof"])
        samples = by_cls["real"][:tgt_real] + by_cls["spoof"][:tgt_spoof]
        rng.shuffle(samples)

    splits = _stratified_split(samples, val_ratio=args.val, test_ratio=args.test, seed=args.seed)

    # Create output dirs
    for split in ("train", "val", "test"):
        if split == "test" and args.test <= 0:
            continue
        for label in ("real", "spoof"):
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_failed = 0
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
            if args.imgsz and args.imgsz > 0:
                im = im.resize((args.imgsz, args.imgsz), resample=Image.BICUBIC)
            _save_jpeg(im, out_path)
            n_written += 1
            written_by_split_class[(split, s.label)] += 1
        except Exception:
            n_failed += 1

    summary = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "samples_total": len(samples),
        "written": n_written,
        "failed": n_failed,
        "max_total": args.max_total,
        "balance": bool(args.balance),
        "val_ratio": args.val,
        "test_ratio": args.test,
        "imgsz": args.imgsz,
        "seed": args.seed,
    }
    print("DONE")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Class balance summary (helps spot imbalance quickly)
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

