import cv2
import numpy as np
import os
from datetime import datetime

CAMERA_INDEX = 2
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
UPSCALE_FACTOR = 2

save_folder = "snapshots"
os.makedirs(save_folder, exist_ok=True)

# -----------------------------
# Open Camera
# -----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Feed", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

# -----------------------------
# Helper Functions
# -----------------------------
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
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# -----------------------------
# Main Loop
# -----------------------------
print("Press 's' to scan document")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # -----------------------------
        # Better Document Detection
        # -----------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edged = cv2.Canny(gray, 50, 150)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        document_contour = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                document_contour = approx
                break

        if document_contour is None:
            print("❌ No document detected.")
            continue

        warped = four_point_transform(frame, document_contour.reshape(4, 2))

        # -----------------------------
        # 🔥 Advanced Enhancement Pipeline
        # -----------------------------

        # Convert to grayscale
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Remove shadows via normalization
        dilated = cv2.dilate(warped_gray, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        shadow_removed = cv2.absdiff(warped_gray, bg)
        shadow_removed = 255 - shadow_removed

        # CLAHE for contrast correction
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(shadow_removed)

        # Denoise (keeps edges sharp)
        denoised = cv2.fastNlMeansDenoising(contrast, h=12)

        # Sharpen
        blur = cv2.GaussianBlur(denoised, (0,0), 3)
        sharpened = cv2.addWeighted(denoised, 1.6, blur, -0.6, 0)

        # Upscale for OCR
        upscaled = cv2.resize(
            sharpened,
            None,
            fx=UPSCALE_FACTOR,
            fy=UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC
        )

        # Clean binarization
        scanned = cv2.adaptiveThreshold(
            upscaled,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10
        )

        # Light morphology cleanup
        kernel = np.ones((2,2), np.uint8)
        scanned = cv2.morphologyEx(scanned, cv2.MORPH_CLOSE, kernel)

        # -----------------------------
        # Save Outputs
        # -----------------------------
        cv2.imwrite(os.path.join(save_folder, f"{timestamp}_color.png"), warped)
        cv2.imwrite(os.path.join(save_folder, f"{timestamp}_enhanced.png"), sharpened)
        cv2.imwrite(os.path.join(save_folder, f"{timestamp}_ocr_ready.png"), scanned)

        print(f"✅ Saved enhanced document set at {timestamp}")
        cv2.imshow("Scanned", scanned)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()