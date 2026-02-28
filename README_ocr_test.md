# ocr_test.py

Batch OCR for images in `snapshots/` using Tesseract and OpenCV.

## What It Does
- Scans `snapshots/` for image files (`.png`, `.jpg`, `.jpeg`).
- Converts each image to grayscale.
- Runs Tesseract OCR and writes a `.txt` file for each image.

## Requirements
- Python 3.8+
- `opencv-python`
- `pytesseract`
- Tesseract OCR engine installed on the system

Install Python deps:
```bash
pip install opencv-python pytesseract
```

Install Tesseract:
- Windows: install from the official Tesseract installer and update the path in `ocr_test.py`.
- macOS: `brew install tesseract`
- Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`

## Configure Tesseract Path (Windows)
Edit in `ocr_test.py` if needed:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Run
From the repo root:
```bash
python ocr_test.py
```

## Output
Results are saved to `ocr_output/` (created automatically):
- For each image `name.png`, a text file `name.png.txt` is written.

## OCR Settings
In `ocr_test.py`:
- `--oem 3`: LSTM OCR engine
- `--psm 6`: assume a uniform block of text

