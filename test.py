import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import datetime
import os

# ----------------------------
# Camera Setup
# ----------------------------
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

os.makedirs("scans", exist_ok=True)

captured_scan = None

# ----------------------------
# Warp + Detection Functions
# ----------------------------
def reorder(points):
    points = points.reshape((4,2))
    newPoints = np.zeros((4,1,2), dtype=np.int32)

    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints

def getBiggestContour(img):
    biggest = np.array([])
    max_area = 0

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest

def warpImage(img, points):
    points = reorder(points)
    pts = points.reshape(4,2)

    width = int(max(
        np.linalg.norm(pts[0]-pts[1]),
        np.linalg.norm(pts[2]-pts[3])
    ))

    height = int(max(
        np.linalg.norm(pts[0]-pts[2]),
        np.linalg.norm(pts[1]-pts[3])
    ))

    dst = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(np.float32(points), dst)
    warped = cv2.warpPerspective(img, matrix, (width, height))

    return warped

def enhanceColorNatural(img):
    img = cv2.bilateralFilter(img, 7, 50, 50)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ----------------------------
# Tkinter UI
# ----------------------------
root = tk.Tk()
root.title("Document Scanner")

label = tk.Label(root)
label.pack()

def save_scan():
    global captured_scan
    if captured_scan is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"scans/scan_{timestamp}.jpg"
        cv2.imwrite(path, captured_scan)
        print("Saved:", path)

save_button = tk.Button(root, text="Save Scan", command=save_scan)
save_button.pack()

def update_frame():
    global captured_scan

    ret, frame = cap.read()
    if not ret:
        return

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    img_dil = cv2.dilate(img_canny, kernel, iterations=2)

    biggest = getBiggestContour(img_dil)

    if biggest.size != 0:
        cv2.drawContours(frame, biggest, -1, (0,255,0), 3)
        warped = warpImage(frame, biggest)
        captured_scan = enhanceColorNatural(warped)

    # Convert for Tkinter
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    label.imgtk = imgtk
    label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()