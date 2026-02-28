from paddleocr import PaddleOCR
import cv2

# Initialize OCR (loads model once)
ocr = PaddleOCR(
   use_textline_orientation=True,   # detects rotated text
    lang='en'             # change if needed
)

image_path = "snapshots/20260228_194103_enhanced.png"

# Run OCR
results = ocr.predict(image_path)

print("\n🔍 Extracted Text:\n")

for line in results[0]:
    text = line[1][0]
    confidence = line[1][1]
    print(f"{text}  ({confidence:.2f})")