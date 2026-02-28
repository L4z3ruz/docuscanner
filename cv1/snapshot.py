import cv2
import numpy as np
import os
from datetime import datetime

CAMERA_INDEX = 2
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
UPSCALE_FACTOR = 2  # Upscale after perspective transform

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
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
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
        # Document Detection
        # -----------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
        # OCR-Ready Preprocessing
        # -----------------------------
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Upscale for better OCR
        warped_gray = cv2.resize(
            warped_gray,
            None,
            fx=UPSCALE_FACTOR,
            fy=UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC
        )

        # Denoise (Non-local Means)
        denoised = cv2.fastNlMeansDenoising(warped_gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # Optional sharpening
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Adaptive threshold for clean black & white
        scanned = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # Save scanned image
        output_path = os.path.join(save_folder, f"scanned_{timestamp}.png")
        cv2.imwrite(output_path, scanned)

        print(f"✅ Document scanned and saved: {output_path}")
        cv2.imshow("Scanned", scanned)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()