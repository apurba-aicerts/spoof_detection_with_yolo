#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    ap.add_argument("--source", type=Path, default=Path("test_image"), help="Folder of images")
    ap.add_argument(
        "--spoof_dir",
        type=Path,
        default=Path("runs/spoof_cls/spoof_test"),
        help="Where to save images predicted as spoof (relative to repo by default)",
    )
    ap.add_argument(
        "--spoof_name",
        type=str,
        default="spoof",
        help="Class name to treat as spoof (must match your dataset folder name)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    weights = (repo_root / args.weights).resolve() if not args.weights.is_absolute() else args.weights.resolve()
    source = (repo_root / args.source).resolve() if not args.source.is_absolute() else args.source.resolve()
    spoof_dir = (repo_root / args.spoof_dir).resolve() if not args.spoof_dir.is_absolute() else args.spoof_dir.resolve()

    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        verbose=False,
    )

    counts: Counter[str] = Counter()
    total = 0
    saved_spoof = 0

    # Save spoof images (copy originals) into spoof_dir
    import shutil

    spoof_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        # Classification: r.probs.top1 is class index; r.names maps idx->name
        if getattr(r, "probs", None) is None:
            continue
        idx = int(r.probs.top1)
        name = r.names.get(idx, str(idx))
        counts[name] += 1
        total += 1

        if str(name).lower() == args.spoof_name.lower():
            src_path = Path(getattr(r, "path", ""))
            if src_path.exists():
                # Avoid collisions: keep original filename, and if duplicates, add a suffix.
                dst = spoof_dir / src_path.name
                if dst.exists():
                    dst = spoof_dir / f"{src_path.stem}__{saved_spoof}{src_path.suffix}"
                shutil.copy2(src_path, dst)
                saved_spoof += 1

    print(f"weights: {weights}")
    print(f"source: {source}")
    print(f"total_images_predicted: {total}")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k}: {v}")
    print(f"spoof_saved_dir: {spoof_dir}")
    print(f"spoof_saved_count: {saved_spoof}")


if __name__ == "__main__":
    main()

