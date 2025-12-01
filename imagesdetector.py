import cv2
import requests
import numpy as np
import sys
import os

API_URL = "http://localhost:5000/predict"
CONF_THRESHOLD = 0.6    # ความมั่นใจขั้นต่ำ
MIN_AREA = 20 * 20      # พื้นที่ contour ขั้นต่ำกัน noise

# ฟังก์ชันตรวจจับตัวอักษรในภาพนิ่ง
def detect_chars_in_image(image_path, show_result=True, save_path=None):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}") #หาไม่พบไฟล์
        return

    # อ่านภาพ
    image = cv2.imread(image_path) #อ่านภาพจากไฟล์
    if image is None:
        print("[ERROR] Cannot read image.")
        return

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # เบลอเล็กน้อยให้เนียนขึ้น
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold เพื่อให้ได้ตัวอักษรสีขาวบนพื้นดำ (หรือกลับกัน)
    _, th = cv2.threshold(
        gray_blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # หา contour ทั้งหมด
    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_AREA:
            continue  # ข้ามชิ้นส่วนที่เล็กเกินไป

        roi = orig[y:y+h, x:x+w].copy()
        ok, buf = cv2.imencode(".jpg", roi)
        if not ok:
            continue

        try:
            r = requests.post(
                API_URL,
                files={"file": buf.tobytes()},
                timeout=1.0
            )
            if not r.ok:
                continue

            data = r.json()
            label = data.get("label", "")
            conf = float(data.get("confidence", 0.0))

            if conf >= CONF_THRESHOLD and label:
                detections.append((x, y, w, h, label, conf))

        except Exception as e:
            print(f"[WARN] API error: {e}")
            continue

    # วาดกรอบและ label ลงบนภาพ
    for (x, y, w, h, label, conf) in detections:
        cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{label} {conf*100:.1f}%"
        cv2.putText(
            orig, text, (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    # แสดงหรือเซฟผลลัพธ์
    if save_path:
        cv2.imwrite(save_path, orig)
        print(f"[INFO] Saved result to {save_path}")

    if show_result:
        cv2.imshow("Character Detection (Image)", orig)
        print("[INFO] Press any key to close window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # พิมพ์ผลลัพธ์รวม
    if detections:
        print("[RESULT] Detections:")
        for (x, y, w, h, label, conf) in detections:
            print(f"  {label} ({conf*100:.1f}%) at x={x}, y={y}, w={w}, h={h}")
    else:
        print("[RESULT] No confident characters found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python imagesdetector.py test1.png")
        sys.exit(1)

    image_path = sys.argv[1]
    # แก้ save_path ได้ตามต้องการ หรือปล่อย None ถ้าไม่อยากเซฟ
    detect_chars_in_image(image_path, show_result=True, save_path=None)
