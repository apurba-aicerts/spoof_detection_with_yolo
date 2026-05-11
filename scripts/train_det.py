#!/usr/bin/env python3
"""Train a YOLO *detection* model for real vs fake face detection."""

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
    ap.add_argument("--data", type=Path, default=Path("datasets/spoof_face_det_mp/dataset.yaml"), help="dataset.yaml path")
    ap.add_argument("--model", type=str, default="yolo26n.pt", help="Pretrained detection checkpoint (e.g. yolo26n.pt)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
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
    ap.add_argument("--out", type=Path, default=Path("runs/spoof_face_det"), help="Output folder (saved inside repo)")
    ap.add_argument("--name", type=str, default="exp")
    args = ap.parse_args()

    data = (repo_root / args.data).resolve() if not args.data.is_absolute() else args.data.resolve()
    if not data.exists():
        raise SystemExit(f"dataset.yaml not found: {data}")

    model = YOLO(args.model)
    device = _resolve_device(args.device)
    out_dir = (repo_root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    model.train(
        data=str(data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        device=device,
        batch=int(args.batch),
        workers=int(args.workers),
        patience=10,
        amp=True,
        val=True,  # keep behavior aligned with classification script's stability defaults
        project=str(out_dir),
        name=str(args.name),
    )


if __name__ == "__main__":
    main()

