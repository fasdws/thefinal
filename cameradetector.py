import cv2
import requests
import numpy as np
import time

# URL ของ API ที่จะรับภาพย่อย (ROI) เพื่อทำนายตัวอักษร
API_URL = "http://localhost:5000/predict"

# --- การตั้งค่ากล้อง ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- พารามิเตอร์ปรับแต่ง ---
CONF_THRESHOLD = 0.6          # วาดกรอบเฉพาะเมื่อ confidence >= ค่านี้
MIN_AREA = 25 * 25            # กรอง contour เล็ก ๆ (ลด noise)
MAX_AREA = 800 * 800          # กรอง contour ใหญ่เกินไป (ถ้าไม่ต้องการเอาออกได้)
MAX_CONTOURS = 8              # จำนวน contour สูงสุดที่จะตรวจ/ส่ง API ต่อเฟรม
API_FRAME_SKIP = 4            # ยิง API ทุก N เฟรม (ยิ่งใหญ่ = โหลดน้อย)

# ตัวแปรช่วยคำนวณ FPS แบบง่าย
frame_id = 0
fps = 0.0
frame_count = 0
fps_start = time.time()
FPS_UPDATE_INTERVAL = 0.5     # อัปเดตแสดง FPS ทุก 0.5 วินาที

# cache ผลลัพธ์ล่าสุด เพื่อใช้ในเฟรมที่ไม่ได้เรียก API
last_detections = []


# --- main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    frame_count += 1

    # --- เตรียมภาพสำหรับหาขอบเขต (contours) ---
    # แปลงเป็นเกรย์, เบลอเล็กน้อย แล้ว threshold โดย Otsu
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # ใช้ THRESH_BINARY_INV เพราะตัวอักษรมักเป็นสีเข้มบนพื้นสว่าง
    _, th = cv2.threshold(
        gray_blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # หาคอนทัวร์ภายนอกทั้งหมด
    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # เลือกเฉพาะ contours ขนาดใหญ่ที่สุด (เพื่อประหยัดการเรียก API)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[:MAX_CONTOURS]

    detections = []

    # เงื่อนไขว่าจะเรียก API ในเฟรมนี้หรือไม่ (ข้ามบางเฟรมเพื่อลดโหลด)
    do_api = (frame_id % API_FRAME_SKIP == 0)

    if do_api:
        # สำหรับแต่ละ contour ให้ตัด bounding box แล้วส่ง ROI ให้ API ทำนาย
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            area = w_box * h_box
            # กรองขนาดที่ไม่ต้องการ
            if area < MIN_AREA or area > MAX_AREA:
                continue

            # ตัด ROI จากเฟรมต้นฉบับ
            roi = frame[y:y+h_box, x:x+w_box].copy()
            ok, buf = cv2.imencode(".jpg", roi)
            if not ok:
                continue

            try:
                # ส่งภาพย่อยไปให้ API เพื่อทำนาย label และ confidence
                r = requests.post(
                    API_URL,
                    files={"file": buf.tobytes()},
                    timeout=0.15  # ควบคุม timeout ไม่ให้ FPS ตก
                )
                if not r.ok:
                    continue

                data = r.json()
                label = data.get("label", "")
                conf = float(data.get("confidence", 0.0))

                # ถ้าผลมั่นใจพอ (>= threshold) ให้เก็บไว้สำหรับวาดกรอบ
                if conf >= CONF_THRESHOLD and label:
                    detections.append((x, y, w_box, h_box, label, conf))

            except Exception:
                # ถ้าเกิด error (timeout, connection) ข้าม contour นี้ไป
                continue

        # เก็บผลล่าสุดไว้ใช้ในเฟรมที่ไม่ได้เรียก API
        last_detections = detections
    else:
        # ถ้าไม่ได้เรียก API ในเฟรมนี้ ให้ใช้ผลเก่าสุด (cache)
        detections = last_detections

    # --- วาดกรอบและป้ายข้อความบนเฟรม ---
    for (x, y, w_box, h_box, label, conf) in detections:
        # วาดกรอบสีเขียวรอบ ROI
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        # ป้ายข้อความแสดง label และเปอร์เซ็นต์ความมั่นใจ
        text = f"{label} {conf*100:.1f}%"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- คำนวณและแสดง FPS แบบง่าย ---
    elapsed = time.time() - fps_start
    if elapsed >= FPS_UPDATE_INTERVAL:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # แสดงผลลัพธ์ในหน้าต่าง OpenCV
    cv2.imshow("Char Detection (High FPS)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ปล่อย resource เมื่อต้องการปิด
cap.release()
cv2.destroyAllWindows()
