import cv2
import numpy as np

# ----------------------------
# Camera setup
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ----------------------------
# Trackbars
# ----------------------------
def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 400, 120)
cv2.createTrackbar("Min Area", "Parameters", 13000, 50000, empty)

# ----------------------------
# Utility functions
# ----------------------------
def reorder(points):
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), np.int32)

    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints


def getBiggestContour(img, minArea):
    biggest = np.array([])
    maxArea = 0

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                # Compute bounding box
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = max(w, h) / min(w, h)
                # Accept if aspect ratio roughly matches A4 (~1.414)
                if 1.3 <= aspect_ratio <= 1.5:
                    if area > maxArea:
                        biggest = approx
                        maxArea = area

    return biggest


def warpImageDynamic(img, points):
    points = reorder(points)
    pts = points.reshape(4, 2)

    widthA = np.linalg.norm(pts[0] - pts[1])
    widthB = np.linalg.norm(pts[2] - pts[3])
    width = int(max(widthA, widthB))

    heightA = np.linalg.norm(pts[0] - pts[2])
    heightB = np.linalg.norm(pts[1] - pts[3])
    height = int(max(heightA, heightB))

    dst = np.float32([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ])

    matrix = cv2.getPerspectiveTransform(np.float32(points), dst)
    warped = cv2.warpPerspective(img, matrix, (width, height))

    return warped


def enhanceColorSharp(img):
    # Convert to LAB for contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 6, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Optional slight denoising
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

    return enhanced

# ----------------------------
# Main loop
# ----------------------------
documentCaptured = False  # lock the document once grabbed
capturedScan = None       # stores the frozen scan

while True:
    success, img = cap.read()
    if not success:
        break

    areaMin = cv2.getTrackbarPos("Min Area", "Parameters")

    imgBlur = cv2.GaussianBlur(img, (5, 5), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    imgDil = cv2.erode(imgDil, kernel, iterations=1)

    cv2.imshow("Camera Feed", img)
    cv2.imshow("Edges", imgDil)

    # Only capture document if not yet captured
    if not documentCaptured:
        biggest = getBiggestContour(imgDil, areaMin)
        if biggest.size != 0:
            imgWarp = warpImageDynamic(img, biggest)
            capturedScan = enhanceColorSharp(imgWarp)
            documentCaptured = True  # lock it

    # Display frozen scan
    if capturedScan is not None:
        cv2.namedWindow("Scanned Document", cv2.WINDOW_NORMAL)
        max_w, max_h = 900, 1200
        scale = min(max_w / capturedScan.shape[1], max_h / capturedScan.shape[0], 1)
        display_w = int(capturedScan.shape[1] * scale)
        display_h = int(capturedScan.shape[0] * scale)
        cv2.imshow("Scanned Document", capturedScan)
        cv2.resizeWindow("Scanned Document", display_w, display_h)

    # Handle keypress
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset scan to grab a new document
        documentCaptured = False
        capturedScan = None
        print("Ready to scan again!")

cap.release()
cv2.destroyAllWindows()
