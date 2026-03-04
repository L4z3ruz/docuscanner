from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException
import cv2
import numpy as np
import pytesseract
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError
import base64
import os
from datetime import datetime

app = Flask(__name__)
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)
API_ROUTES = {"/detect", "/scan", "/save"}

# Allow explicit OCR binary path override, with a Windows default fallback.
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
else:
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

# ─── Geometry helpers ────────────────────────────────────────────────────────

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    maxWidth  = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxHeight = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0],[maxWidth-1, 0],[maxWidth-1, maxHeight-1],[0, maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# ─── Detection ───────────────────────────────────────────────────────────────

def detect_document_contour(frame):
    """
    Multi-strategy detector tuned for laptop cameras.
    Returns (4,2) float32 array or None.
    """
    h, w = frame.shape[:2]
    img_area = h * w

    strategies = [
        # (blur_ksize, canny_lo, canny_hi, approx_eps)
        (5, 30, 100, 0.02),
        (7, 50, 150, 0.02),
        (9, 10, 80,  0.03),
        (5, 80, 200, 0.02),
    ]

    for bk, clo, chi, eps in strategies:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (bk, bk), 0)
        edged = cv2.Canny(gray, clo, chi)
        edged = cv2.dilate(edged, None, iterations=2)
        edged = cv2.erode(edged, None, iterations=1)

        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        for c in cnts:
            area = cv2.contourArea(c)
            if area < img_area * 0.05:          # ignore tiny blobs
                continue
            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype("float32")

    return None

# ─── Enhancement pipeline ────────────────────────────────────────────────────

def enhance(warped, mode="scan"):
    """Return (color_png_bytes, bw_png_bytes)."""
    color_bytes = cv2.imencode(".png", warped)[1].tobytes()

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Shadow removal
    dil = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg  = cv2.medianBlur(dil, 21)
    sr  = cv2.absdiff(gray, bg)
    sr  = 255 - sr

    # CLAHE
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast = clahe.apply(sr)

    # Denoise + sharpen
    denoised = cv2.fastNlMeansDenoising(contrast, h=10)
    blurred  = cv2.GaussianBlur(denoised, (0,0), 3)
    sharp    = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    # Upscale 2×
    up = cv2.resize(sharp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if mode == "scan":
        final = cv2.adaptiveThreshold(up, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        kern  = np.ones((2,2), np.uint8)
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kern)
    else:
        final = up   # grayscale enhanced, no binarisation

    bw_bytes = cv2.imencode(".png", final)[1].tobytes()
    return color_bytes, bw_bytes, final

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    """Return the detected corner points for live preview overlay."""
    payload = request.get_json(silent=True) or {}
    data  = payload.get("image", "")
    img   = _b64_to_cv2(data)
    if img is None:
        return jsonify({"found": False})

    pts = detect_document_contour(img)
    if pts is None:
        return jsonify({"found": False})

    h, w = img.shape[:2]
    norm = (pts / [w, h]).tolist()   # normalised 0-1 for the JS overlay
    return jsonify({"found": True, "points": norm})

@app.route("/scan", methods=["POST"])
def scan():
    """Full scan: warp + enhance + OCR + save."""
    payload   = request.get_json(silent=True) or {}
    data      = payload.get("image", "")
    scan_mode = payload.get("mode", "scan")   # "scan" | "photo"

    img = _b64_to_cv2(data)
    if img is None:
        return jsonify({"ok": False, "error": "Bad image data"})

    pts = detect_document_contour(img)
    if pts is None:
        # Fall back: use entire frame
        h, w = img.shape[:2]
        pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype="float32")

    warped = four_point_transform(img, pts)
    color_bytes, bw_bytes, bw_mat = enhance(warped, mode=scan_mode)
    scanned_bytes = color_bytes if scan_mode == "photo" else bw_bytes

    # OCR (non-fatal: scan can succeed even if OCR tooling is unavailable).
    ocr_text = ""
    warning = None
    try:
        ocr_text = pytesseract.image_to_string(bw_mat, config="--psm 6").strip()
    except TesseractNotFoundError:
        warning = "OCR skipped: Tesseract is not installed or not in PATH."
    except TesseractError as e:
        warning = f"OCR skipped: {str(e)}"

    # Save
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    color_path = os.path.join(SAVE_DIR, f"{ts}_color.png")
    bw_path    = os.path.join(SAVE_DIR, f"{ts}_scanned.png")
    cv2.imwrite(color_path, warped)
    open(bw_path, "wb").write(scanned_bytes)

    return jsonify({
        "ok":        True,
        "color":     _to_b64(color_bytes),
        "scanned":   _to_b64(scanned_bytes),
        "ocr":       ocr_text,
        "filename":  ts,
        "warning":   warning,
    })

@app.route("/outputs/<path:fname>")
def outputs(fname):
    return send_from_directory(SAVE_DIR, fname)

# Save results from the frontend (color/scanned images + OCR text).
@app.route("/save", methods=["POST"])
def save():
    payload = request.get_json(silent=True) or {}
    color_b64 = payload.get("color", "")
    scanned_b64 = payload.get("scanned", "")
    ocr_text = payload.get("ocr", "")
    filename = payload.get("filename") or datetime.now().strftime("%Y%m%d_%H%M%S")

    if not color_b64 or not scanned_b64:
        return jsonify({"ok": False, "error": "Missing image data"})

    color_bytes = _b64_to_bytes(color_b64)
    scanned_bytes = _b64_to_bytes(scanned_b64)
    if color_bytes is None or scanned_bytes is None:
        return jsonify({"ok": False, "error": "Bad image data"})

    color_path = os.path.join(SAVE_DIR, f"{filename}_color.png")
    scan_path = os.path.join(SAVE_DIR, f"{filename}_scanned.png")
    text_path = os.path.join(SAVE_DIR, f"{filename}_ocr.txt")

    with open(color_path, "wb") as f:
        f.write(color_bytes)
    with open(scan_path, "wb") as f:
        f.write(scanned_bytes)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(ocr_text or "")

    return jsonify({
        "ok": True,
        "color": f"/outputs/{os.path.basename(color_path)}",
        "scanned": f"/outputs/{os.path.basename(scan_path)}",
        "ocr": f"/outputs/{os.path.basename(text_path)}",
    })

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _b64_to_cv2(b64str):
    try:
        b64str = b64str.split(",")[-1]
        arr    = np.frombuffer(base64.b64decode(b64str), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def _b64_to_bytes(b64str):
    try:
        b64str = b64str.split(",")[-1]
        return base64.b64decode(b64str)
    except Exception:
        return None

def _to_b64(raw_bytes):
    return "data:image/png;base64," + base64.b64encode(raw_bytes).decode()

@app.errorhandler(Exception)
def handle_exception(err):
    if request.path in API_ROUTES:
        app.logger.exception("API error on %s", request.path)
        return jsonify({"ok": False, "error": str(err)}), 500
    if isinstance(err, HTTPException):
        return err
    return "Internal Server Error", 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
