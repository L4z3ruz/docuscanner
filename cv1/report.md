## Abstract
This project presents a web-based document scanner and OCR application developed with a Flask backend (`app.py`) and a single-page frontend (`index.html`). The system is designed to convert raw camera captures into clean digital documents through an automated pipeline that includes document boundary detection, perspective correction, image enhancement, text extraction, and output storage. It supports both live webcam input and still-image upload, making it suitable for laptop-based scanning without dedicated scanner hardware.  
From the user side, the frontend provides real-time visual feedback through a contour overlay, scan mode selection for B&W or color-oriented output, immediate result preview, OCR text display, and export features for JPG/PDF. From the system side, OpenCV-based preprocessing improves readability under common real-world issues such as uneven lighting and mild blur, while Tesseract OCR extracts machine-readable text for downstream use. Overall, the application demonstrates a practical, low-cost approach to document digitization by combining computer vision methods with browser-based usability.

## Intro
Digitizing physical documents is now a routine requirement in education, administration, and business workflows. Users often need to capture notes, forms, receipts, and printed pages quickly, then store them in searchable and shareable formats. While flatbed scanners can deliver high-quality results, they are less accessible in day-to-day environments where users primarily rely on webcams or phone images. Raw camera captures usually contain perspective distortion, background clutter, shadows, and inconsistent contrast, which reduce readability and OCR accuracy if left unprocessed.  
This project addresses these limitations by implementing a complete browser-based document scanning workflow that automates the most important preprocessing steps. The system detects likely document boundaries, rectifies the page to a top-down view, applies enhancement operations to improve clarity, and then performs OCR for text extraction. The architecture combines OpenCV and Tesseract in a Flask backend with a responsive JavaScript frontend, exposing functionality through simple REST endpoints. As a result, the application provides both a technically meaningful computer vision pipeline and a user-friendly interface suitable for practical scanning tasks.

## Aims and Objectives
1. Develop an end-to-end document scanning web app using Python Flask and JavaScript.
2. Detect rectangular document contours robustly from noisy camera frames.
3. Apply perspective transformation to produce top-down, flattened scans.
4. Improve scan readability using enhancement steps such as shadow removal, CLAHE, denoising, sharpening, and thresholding.
5. Integrate OCR text extraction from scanned outputs.
6. Support both live camera capture and still-image upload.
7. Provide export and persistence features (saved images, OCR text, JPG/PDF download).

## Theory (methods ???)
### 1. Document Detection
- Frames are converted to grayscale, blurred, and edge-detected using Canny.
- Morphological operations (dilation/erosion) strengthen edge continuity.
- External contours are extracted and sorted by area.
- Quadrilateral approximation (`approxPolyDP`) identifies likely document boundaries.
- Multiple parameter strategies are tried to improve robustness under varying lighting/background conditions.

### 2. Geometric Rectification
- Detected corner points are ordered (top-left, top-right, bottom-right, bottom-left).
- A perspective transform matrix is computed.
- `warpPerspective` maps the skewed document to a rectangular, front-facing scan.

### 3. Image Enhancement
Pipeline used in `enhance()`:
1. Grayscale conversion
2. Shadow suppression via background estimation (dilate + median blur + absolute difference)
3. Local contrast improvement using CLAHE
4. Denoising and sharpening
5. 2x upscaling for better OCR resolution
6. Adaptive thresholding + morphological closing (in scan mode)

Two output styles are supported:
- `scan` mode: binarized B&W document style
- `photo` mode: enhanced grayscale/photo style

### 4. OCR
- OCR is performed with `pytesseract.image_to_string(..., --psm 6)`.
- OCR failure is handled gracefully (scan still succeeds with warning).
- Text output is shown in UI and optionally saved as `.txt`.

## System implementation
### Backend (`app.py`)
Main routes:
1. `/` serves the frontend template.
2. `/detect` accepts base64 image input, detects document contour, returns normalized corner points for overlay.
3. `/scan` performs full pipeline (detect -> warp -> enhance -> OCR -> save), returns base64 results and OCR text.
4. `/save` persists color scan, processed scan, and OCR text to disk.
5. `/outputs/<fname>` serves saved files.

Implementation characteristics:
- Flask API + HTML template rendering.
- OpenCV for CV operations.
- Tesseract path auto-detection with optional `TESSERACT_CMD`.
- Centralized API error handling for JSON responses.
- Output directory auto-created (`outputs`).

### Frontend (`index.html`)
Core UI modules:
1. Live camera preview with device selector (`getUserMedia` + `enumerateDevices`).
2. Real-time detection polling (`/detect`) and canvas polygon overlay.
3. Scan controls (mode toggle, capture, file upload, clear/reset).
4. Result tab with preview image, OCR display, copy function, download controls.
5. Download conversion to JPG/PDF via canvas + jsPDF.
6. History tab storing recent scan sessions in-memory.
7. Auto-save call to backend (`/save`) after successful scan.

Interaction flow:
1. User starts camera or selects image.
2. Frontend sends frame/image to `/detect` for boundary preview.
3. User triggers scan; frontend posts to `/scan`.
4. Backend returns processed image + OCR.
5. Frontend displays result, stores in history, and saves to server via `/save`.

## Conclusion
The application successfully implements a complete browser-based document scanning pipeline using practical, lightweight technologies. It provides real-time detection feedback, produces corrected/enhanced scan outputs, and extracts useful text with OCR. The architecture is simple and modular enough for academic demonstration and practical small-scale use. Overall, the project meets its goals of accessibility, automation, and functional scan quality.

## Further extension
1. Add manual corner adjustment UI when auto-detection fails.
2. Support multi-page scanning and combined PDF export.
3. Add OCR language selection and mixed-language support.
4. Store history in database (not only in-memory UI state).
5. Add authentication and per-user document libraries.
6. Improve performance with async/background OCR queue for large images.
7. Add scan quality metrics (blur detection, skew confidence, OCR confidence).
8. Add unit/integration tests for CV pipeline and API contract.
9. Provide Docker deployment and production-ready config (gunicorn, logging, monitoring).

## Reference
1. Bradski, G. "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.
2. OpenCV Documentation: https://docs.opencv.org/
3. Flask Documentation: https://flask.palletsprojects.com/
4. Tesseract OCR Documentation: https://tesseract-ocr.github.io/
5. PyTesseract Documentation: https://pypi.org/project/pytesseract/
6. jsPDF Documentation: https://github.com/parallax/jsPDF
