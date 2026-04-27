#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="Path to trained weights (e.g. runs/.../best.pt)")
    ap.add_argument("--format", type=str, default="onnx", help="onnx, openvino, engine, torchscript, ...")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    weights = args.weights.resolve()
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    model.export(format=args.format, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()

