"""
Windows client: send webcam frames to the WSL API.

Install on Windows:
  pip install opencv-python requests

Run:
  python windows_webcam_client.py --api http://127.0.0.1:8011
"""

from __future__ import annotations

import argparse
import time

import cv2
import requests


def _predict_url(api: str, url: str) -> tuple[str, float]:
    r = requests.post(f"{api}/predict/url?device=cpu", json={"url": url}, timeout=15)
    r.raise_for_status()
    j = r.json()
    return str(j.get("label")), float(j.get("confidence", 0.0))


def _predict_frame(api: str, frame) -> tuple[str, float]:
    ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok_enc:
        raise RuntimeError("Failed to encode frame.")
    files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    r = requests.post(f"{api}/predict/upload?device=cpu", files=files, timeout=5)
    r.raise_for_status()
    j = r.json()
    return str(j.get("label")), float(j.get("confidence", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", type=str, default="http://127.0.0.1:8011")
    ap.add_argument("--mode", type=str, default="webcam", choices=["webcam", "url"], help="webcam (default) or url")
    ap.add_argument("--url", type=str, default="", help="If set, run a single prediction on this image URL and exit")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--every_ms", type=int, default=200, help="Send one frame every N ms")
    args = ap.parse_args()

    # URL mode (interactive or single-shot)
    if args.mode == "url":
        if args.url:
            label, conf = _predict_url(args.api, args.url)
            print(f"{label} {conf:.4f}")
            return

        print("Enter image URL (or 'q' to quit).")
        while True:
            try:
                url = input("> ").strip()
            except KeyboardInterrupt:
                print()
                return
            if not url or url.lower() in {"q", "quit", "exit"}:
                return
            try:
                label, conf = _predict_url(args.api, url)
                print(f"{label} {conf:.4f}")
            except Exception as e:
                print(f"Error: {e}")
        return

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    last = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        if (now - last) * 1000 >= args.every_ms:
            last = now
            try:
                label, conf = _predict_frame(args.api, frame)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255) if str(label).lower() == "spoof" else (0, 255, 0),
                    2,
                )
            except Exception as e:
                cv2.putText(frame, f"API error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Spoof Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

