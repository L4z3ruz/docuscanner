import cv2
import pytesseract
import os

# -----------------------------
# If on Windows, set the Tesseract path
# -----------------------------
# Example: C:\Program Files\Tesseract-OCR\tesseract.exe
# Uncomment and edit if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------
# Folder containing scanned documents
# -----------------------------
scanned_folder = "snapshots"
output_folder = "ocr_output"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# OCR Settings
# -----------------------------
custom_config = r'--oem 3 --psm 6'  # OEM 3 = LSTM, PSM 6 = Assume a uniform block of text

# -----------------------------
# Process each scanned document
# -----------------------------
for filename in os.listdir(scanned_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(scanned_folder, filename)
        image = cv2.imread(path)

        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run OCR
        text = pytesseract.image_to_string(gray, config=custom_config)

        # Save extracted text
        text_file = os.path.join(output_folder, filename + ".txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Processed {filename} -> {text_file}")