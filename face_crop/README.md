# `face_crop/` workspace

This folder is a **sandbox for face-crop / bounding-box experiments**.
We keep it separate so we can iterate without touching the main training/API pipeline until decisions are finalized.

## What’s inside

- **`sample/`**: a small copied subset of the dataset for quick iteration
  - `sample/live/` and `sample/spoof/` contain `*.jpg` (+ sometimes `*_BB.txt`)
- **`experiments/`**: experiment scripts (visualization/cropping prototypes)
  - `draw_bb_overlay.py`: draw RetinaFace `*_BB.txt` boxes (scaled from 224-ref) with optional padding
- **`outputs/`**: generated artifacts written by experiments

## Bounding box format (RetinaFace, from `data_set/README`)

`*_BB.txt` contains:

- `x y w h conf`
  - `x, y`: upper-left in a **224×224 reference space**
  - `w, h`: width/height in that same reference space
  - `conf`: detection score

To map boxes to the real image size \((W,H)\):

- \(x' = x \cdot \frac{W}{224}\)
- \(y' = y \cdot \frac{H}{224}\)
- \(w' = w \cdot \frac{W}{224}\)
- \(h' = h \cdot \frac{H}{224}\)

## Draw bbox overlays (RetinaFace)

Live samples:

```bash
./spoof_env/bin/python face_crop/experiments/draw_bb_overlay.py \
  --in_folder face_crop/sample/live \
  --run_name overlay_face_live \
  --ref_size 224 \
  --max_images 30 \
  --width 4
```

Spoof samples:

```bash
./spoof_env/bin/python face_crop/experiments/draw_bb_overlay.py \
  --in_folder face_crop/sample/spoof \
  --run_name overlay_face_spoof \
  --ref_size 224 \
  --max_images 30 \
  --width 4
```

### “Head-ish” overlay (padding experiment)

If you want to see how a padded ROI would look (more forehead/head), try:

```bash
./spoof_env/bin/python face_crop/experiments/draw_bb_overlay.py \
  --in_folder face_crop/sample/live \
  --run_name overlay_headish_live \
  --ref_size 224 \
  --max_images 30 \
  --width 4 \
  --pad 0.20 \
  --pad_top_mult 1.8
```

Outputs are written under `face_crop/outputs/<run_name>/`.

