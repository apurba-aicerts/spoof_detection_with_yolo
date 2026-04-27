from __future__ import annotations

import logging
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "spoof_cls" / "exp-21" / "weights" / "best.pt"

log = logging.getLogger("api")


class PredictURLRequest(BaseModel):
    url: HttpUrl


class PredictResponse(BaseModel):
    label: str
    confidence: float


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _disable_cuda_for_process() -> None:
    """
    Prevent native CUDA probing in environments with broken CUDA stubs/drivers.
    This is intentionally process-global: we run the API in CPU mode.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch  # type: ignore

        # Ultralytics checks torch.cuda.is_available() during init; avoid calling into CUDA.
        if hasattr(torch, "cuda"):
            torch.cuda.is_available = lambda: False  # type: ignore[assignment]
            torch.cuda.device_count = lambda: 0  # type: ignore[assignment]
    except Exception:
        # If torch isn't importable yet, we still keep CUDA_VISIBLE_DEVICES empty.
        pass


def _resolve_device(device: Optional[str]) -> str:
    d = (device or "").strip().lower()
    if d in {"", "auto"}:
        if _env_truthy("API_FORCE_CPU"):
            return "cpu"
        try:
            import torch  # type: ignore

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if d in {"gpu", "cuda"}:
        return "0"
    return device or "cpu"


@lru_cache(maxsize=1)
def get_model(force_cpu: bool = False) -> Any:
    # Import Ultralytics lazily to avoid native crashes during server startup in some environments.
    # (We load the model only on first request.)
    log.info("get_model(): CUDA_VISIBLE_DEVICES=%r", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if force_cpu or _env_truthy("API_FORCE_CPU"):
        _disable_cuda_for_process()
        log.info("get_model(): forcing CPU mode (CUDA_VISIBLE_DEVICES=%r)", os.environ.get("CUDA_VISIBLE_DEVICES"))
    log.info("get_model(): importing ultralytics.YOLO (lazy)")
    from ultralytics import YOLO  # type: ignore

    weights = Path(os.getenv("SPOOF_WEIGHTS", str(DEFAULT_WEIGHTS))).expanduser()
    if not weights.is_absolute():
        weights = (REPO_ROOT / weights).resolve()
    if not weights.exists():
        raise RuntimeError(f"Weights not found: {weights}")
    log.info("get_model(): loading weights %s", weights)
    return YOLO(str(weights))


def pil_from_bytes(data: bytes) -> Image.Image:
    try:
        im = Image.open(BytesIO(data))
        im.load()
        return im.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def predict_pil(im: Image.Image, device: str = "cpu") -> PredictResponse:
    device = _resolve_device(device)
    log.info("predict_pil(): device=%s", device)
    model = get_model(force_cpu=str(device).lower() == "cpu")
    # For classification, Ultralytics returns Results with .probs and .names
    try:
        res = model.predict(im, verbose=False, device=device)[0]
    except Exception as e:
        # If a GPU request fails (bad device string, CUDA hiccup), fall back to CPU.
        if str(device).lower() != "cpu":
            log.warning("predict_pil(): device=%s failed (%s); falling back to cpu", device, e)
            res = model.predict(im, verbose=False, device="cpu")[0]
        else:
            raise
    if getattr(res, "probs", None) is None:
        raise HTTPException(status_code=500, detail="Model did not return classification probabilities.")
    idx = int(res.probs.top1)
    label = str(res.names.get(idx, idx))
    conf = float(res.probs.top1conf)
    log.info("predict_pil(): label=%s conf=%.4f", label, conf)
    return PredictResponse(label=label, confidence=conf)


app = FastAPI(title="Spoof Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    # Keep health lightweight: report whether weights path exists.
    # Model is loaded lazily on first prediction request.
    weights = Path(os.getenv("SPOOF_WEIGHTS", str(DEFAULT_WEIGHTS))).expanduser()
    if not weights.is_absolute():
        weights = (REPO_ROOT / weights).resolve()
    return {"ok": True, "weights_exists": weights.exists(), "weights": str(weights)}


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    file: UploadFile = File(...),
    device: Optional[str] = "cpu",
) -> PredictResponse:
    log.info("/predict/upload: filename=%s content_type=%s device=%s", file.filename, file.content_type, device)
    data = await file.read()
    im = pil_from_bytes(data)
    return predict_pil(im, device=device or "cpu")


@app.post("/predict/url", response_model=PredictResponse)
def predict_url(payload: PredictURLRequest, device: Optional[str] = "cpu") -> PredictResponse:
    log.info("/predict/url: url=%s device=%s", payload.url, device)
    try:
        r = requests.get(str(payload.url), timeout=15)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
    im = pil_from_bytes(r.content)
    return predict_pil(im, device=device or "cpu")

