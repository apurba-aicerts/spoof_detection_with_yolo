#!/usr/bin/env python3
from __future__ import annotations

"""
Draw RetinaFace bounding boxes on images for verification (work ONLY under face_crop/).

BB format (per data_set/README):
  x y w h conf

Important:
- BB coords are stored in a 224x224 reference space.
- Convert to real image space before drawing:
    x' = x * (W / 224)
    y' = y * (H / 224)
    w' = w * (W / 224)
    h' = h * (H / 224)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    conf: float | None = None


def parse_bb_line(line: str) -> Box:
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Bad BB line: {line!r}")
    x, y, w, h = map(float, parts[:4])
    conf = float(parts[4]) if len(parts) >= 5 else None
    return Box(x=x, y=y, w=w, h=h, conf=conf)


def load_boxes(bb_path: Path) -> list[Box]:
    lines = [ln.strip() for ln in bb_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    return [parse_bb_line(ln) for ln in lines]


def scale_from_ref(box: Box, real_w: int, real_h: int, ref_size: int) -> Box:
    if ref_size <= 0:
        return box
    sx = real_w / float(ref_size)
    sy = real_h / float(ref_size)
    return Box(x=box.x * sx, y=box.y * sy, w=box.w * sx, h=box.h * sy, conf=box.conf)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def to_xyxy(box: Box) -> tuple[float, float, float, float]:
    return box.x, box.y, box.x + box.w, box.y + box.h


def expand_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    pad: float,
    pad_top_mult: float,
) -> tuple[float, float, float, float]:
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * float(pad)
    py = bh * float(pad)
    return (x1 - px, y1 - py * float(pad_top_mult), x2 + px, y2 + py)


def iter_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and ":" not in p.name:
            out.append(p)
    return sorted(out)


def _try_load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def draw_overlay(
    im: Image.Image,
    boxes: list[Box],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    width: int = 3,
    show_text: bool = True,
    pad: float = 0.0,
    pad_top_mult: float = 1.0,
) -> Image.Image:
    out = im.copy()
    d = ImageDraw.Draw(out)
    font = _try_load_font(16)

    w_img, h_img = out.size
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = to_xyxy(b)
        if pad and pad > 0:
            x1, y1, x2, y2 = expand_xyxy(x1, y1, x2, y2, pad=float(pad), pad_top_mult=float(pad_top_mult))
        x1 = clamp(x1, 0, w_img - 1)
        y1 = clamp(y1, 0, h_img - 1)
        x2 = clamp(x2, 0, w_img - 1)
        y2 = clamp(y2, 0, h_img - 1)

        d.rectangle([x1, y1, x2, y2], outline=color, width=width)

        if show_text:
            conf_txt = f"{b.conf:.3f}" if b.conf is not None else "n/a"
            label = f"b{i} {conf_txt}"
            tx, ty = int(x1), int(max(0, y1 - 18))
            tw, _ = d.textbbox((tx, ty), label, font=font)[2:]
            d.rectangle([tx, ty, tx + tw + 6, ty + 18], fill=(0, 0, 0))
            d.text((tx + 3, ty + 1), label, fill=(255, 255, 255), font=font)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_folder", type=Path, default=Path("face_crop/sample/live"))
    ap.add_argument("--run_name", type=str, default="overlay_face_ref224", help="Subfolder under face_crop/outputs/")
    ap.add_argument("--ref_size", type=int, default=224, help="Reference size for BB coords (224 per dataset README)")
    ap.add_argument("--max_images", type=int, default=12)
    ap.add_argument("--width", type=int, default=3, help="Line width")
    ap.add_argument("--no_text", action="store_true", help="Disable confidence text")
    ap.add_argument("--pad", type=float, default=0.0, help="Pad ratio around the box (e.g. 0.2 = 20%)")
    ap.add_argument(
        "--pad_top_mult",
        type=float,
        default=1.5,
        help="Multiply top padding to include more forehead/head (only used when --pad > 0)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    in_folder = (repo_root / args.in_folder).resolve() if not args.in_folder.is_absolute() else args.in_folder.resolve()
    out_dir = (repo_root / "face_crop" / "outputs" / args.run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(in_folder)[: max(args.max_images, 0) or None]
    n_ok = 0
    n_fail = 0

    for img_path in images:
        bb_path = img_path.with_name(f"{img_path.stem}_BB.txt")
        if not bb_path.exists():
            n_fail += 1
            print(f"Missing BB: {bb_path}")
            continue

        try:
            im = Image.open(img_path)
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
            boxes = load_boxes(bb_path)
            boxes = [scale_from_ref(b, real_w=im.size[0], real_h=im.size[1], ref_size=args.ref_size) for b in boxes]
            over = draw_overlay(
                im,
                boxes,
                width=args.width,
                show_text=not args.no_text,
                pad=float(args.pad),
                pad_top_mult=float(args.pad_top_mult),
            )
            out_path = out_dir / f"{img_path.stem}__overlay_r{args.ref_size}_pad{args.pad}_top{args.pad_top_mult}.jpg"
            over.save(out_path, format="JPEG", quality=92, optimize=True)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"Failed: {img_path} ({e})")

    print("DONE")
    print(f"in_folder: {in_folder}")
    print(f"out_dir: {out_dir}")
    print(f"ok_images: {n_ok}  failed_images: {n_fail}  total_images: {len(images)}")


if __name__ == "__main__":
    main()

