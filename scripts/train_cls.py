#!/usr/bin/env python3
"""Train an Ultralytics YOLO *classification* model for spoof detection.

Reads a prepared classification dataset (see `scripts/prepare_cls_dataset*.py`)
and runs Ultralytics training, choosing a stable default `--workers` value on
WSL2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _resolve_device(device_arg: str) -> str:
    d = (device_arg or "").strip().lower()
    if d in {"auto", ""}:
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device_arg


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("datasets/spoof_face_cls"), help="Dataset root")
    ap.add_argument("--model", type=str, default="yolo26m-cls.pt", help="Pretrained model (.pt)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", type=str, default="auto", help="auto | 0 | cpu")
    ap.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = Ultralytics AutoBatch)")
    try:
        is_wsl = "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        is_wsl = False
    ap.add_argument(
        "--workers",
        type=int,
        default=0 if is_wsl else 8,
        help="Dataloader workers (0 is most stable on WSL2/low /dev/shm)",
    )
    ap.add_argument("--out", type=Path, default=Path("runs/spoof_face_cls"), help="Output folder (saved inside repo)")
    ap.add_argument("--name", type=str, default="exp")
    args = ap.parse_args()

    data_root = (repo_root / args.data).resolve() if not args.data.is_absolute() else args.data.resolve()
    if not data_root.exists():
        raise SystemExit(f"Dataset not found: {data_root}. Run scripts/prepare_cls_dataset.py first.")

    model = YOLO(args.model)
    device = _resolve_device(args.device)
    out_dir = (repo_root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    model.train(
        data=str(data_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        patience=10,
        batch=args.batch,
        workers=args.workers,
        amp=True,
        val=True,  # avoid torchvision.nms CUDA issues during Ultralytics final_eval
        project=str(out_dir),
        name=args.name,
    )


if __name__ == "__main__":
    main()

