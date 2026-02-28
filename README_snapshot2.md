# snapshot2.py

Live camera document scanner with perspective correction and an OCR-ready enhancement pipeline.

## What It Does
- Opens a camera feed, detects a document contour, and applies a 4‑point perspective transform.
- Enhances the scan (shadow removal, contrast, denoise, sharpen, upscale, adaptive threshold).
- Saves three outputs per scan: color, enhanced, and OCR‑ready images.

## Requirements
- Python 3.8+
- `opencv-python`
- `numpy`

Install:
```bash
pip install opencv-python numpy
```

## Run
From the repo root:
```bash
python cv1/snapshot2.py
```

Controls:
- Press `s` to scan and save a document set.
- Press `q` to quit.

## Output
Files are saved into `snapshots/` (created automatically):
- `YYYYMMDD_HHMMSS_color.png`
- `YYYYMMDD_HHMMSS_enhanced.png`
- `YYYYMMDD_HHMMSS_ocr_ready.png`

## Key Settings
Edit near the top of `cv1/snapshot2.py`:
- `CAMERA_INDEX`: camera device index (default `2`).
- `TARGET_WIDTH`, `TARGET_HEIGHT`: capture resolution.
- `UPSCALE_FACTOR`: scaling before adaptive thresholding.

## Troubleshooting
- If the camera does not open, try `CAMERA_INDEX = 0` or `1`.
- If no document is detected, increase lighting or place the paper against a contrasting background.

