from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CropResult:
    crop_rgb: np.ndarray  # HWC uint8 RGB
    score: float
    xyxy_raw: tuple[int, int, int, int]
    xyxy_padded: tuple[int, int, int, int]


def _mp_box_to_xyxy(img_w: int, img_h: int, rel_box: Any) -> tuple[int, int, int, int]:
    x_center = rel_box.xmin + rel_box.width / 2.0
    y_center = rel_box.ymin + rel_box.height / 2.0
    w = rel_box.width
    h = rel_box.height
    x1 = int((x_center - w / 2.0) * img_w)
    y1 = int((y_center - h / 2.0) * img_h)
    x2 = int((x_center + w / 2.0) * img_w)
    y2 = int((y_center + h / 2.0) * img_h)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return (x1, y1, x2, y2)


def _pick_best(dets: list[tuple[tuple[int, int, int, int], float]]) -> tuple[tuple[int, int, int, int], float] | None:
    if not dets:
        return None

    def key(d: tuple[tuple[int, int, int, int], float]) -> tuple[float, int]:
        (x1, y1, x2, y2), score = d
        area = max(1, x2 - x1) * max(1, y2 - y1)
        return (float(score), int(area))

    return sorted(dets, key=key, reverse=True)[0]


def _pad_xyxy(
    xyxy: tuple[int, int, int, int],
    *,
    img_w: int,
    img_h: int,
    pad: float,
    pad_top_mult: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * float(pad)))
    py = int(round(bh * float(pad)))

    x1p = max(0, x1 - px)
    y1p = max(0, y1 - int(round(py * float(pad_top_mult))))
    x2p = min(img_w, x2 + px)
    y2p = min(img_h, y2 + py)
    return (x1p, y1p, x2p, y2p)


def _crop_rgb(rgb: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = xyxy
    if x2 <= x1 or y2 <= y1:
        return None
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


@lru_cache(maxsize=1)
def _get_mp():
    # Lazy import (so non-MP users can still import this module)
    try:
        import mediapipe as mp  # type: ignore
    except Exception as e:
        raise RuntimeError("MediaPipe is not installed. Install: pip install mediapipe") from e
    return mp


@lru_cache(maxsize=8)
def _get_detector(model_selection: int, min_conf: float):
    # NOTE: Creating FaceDetection repeatedly is very slow/noisy (EGL/TFLite init) and can lead
    # to the process being killed during large dataset prep. Cache detectors across calls.
    mp = _get_mp()
    return mp.solutions.face_detection.FaceDetection(
        model_selection=int(model_selection),
        min_detection_confidence=float(min_conf),
    )


def crop_best_face_mediapipe(
    rgb: np.ndarray,
    *,
    min_conf: float = 0.6,
    model_selection: int = 1,
    pad: float = 0.20,
    pad_top_mult: float = 1.5,
) -> CropResult | None:
    """
    Crop the best face from an RGB image using MediaPipe BlazeFace.

    Lazy-imports `mediapipe` so non-cropping paths can run without it installed.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HWC RGB image")

    det = _get_detector(int(model_selection), float(min_conf))
    res = det.process(rgb)
    h, w = rgb.shape[:2]

    dets: list[tuple[tuple[int, int, int, int], float]] = []
    if res and res.detections:
        for d in res.detections:
            score = float(d.score[0]) if d.score else 0.0
            if score < float(min_conf):
                continue
            rel_box = d.location_data.relative_bounding_box
            xyxy = _mp_box_to_xyxy(w, h, rel_box)
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append((xyxy, score))

    best = _pick_best(dets)
    if best is None:
        return None
    xyxy_raw, score = best
    xyxy_pad = _pad_xyxy(xyxy_raw, img_w=w, img_h=h, pad=float(pad), pad_top_mult=float(pad_top_mult))
    crop = _crop_rgb(rgb, xyxy_pad)
    if crop is None:
        return None
    return CropResult(crop_rgb=crop, score=float(score), xyxy_raw=xyxy_raw, xyxy_padded=xyxy_pad)

