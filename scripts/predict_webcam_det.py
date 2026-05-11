#!/usr/bin/env python3
"""Webcam demo: run a YOLO detection model and overlay real/fake face boxes."""

from __future__ import annotations

import argparse

import cv2
from ultralytics import YOLO


def _color_for(cls_id: int) -> tuple[int, int, int]:
    # BGR
    return (0, 200, 0) if cls_id == 0 else (0, 0, 230)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to trained detection weights (.pt)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(int(args.cam))
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            res = model.predict(
                source=frame_bgr,
                imgsz=int(args.imgsz),
                conf=float(args.conf),
                device=str(args.device),
                verbose=False,
            )
            r0 = res[0]
            boxes = getattr(r0, "boxes", None)
            if boxes is not None:
                for b in boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = _color_for(cls_id)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    label = "real" if cls_id == 0 else "fake"
                    cv2.putText(
                        frame_bgr,
                        f"{label} {conf:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow("spoof face det (real=green, fake=red)", frame_bgr)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

