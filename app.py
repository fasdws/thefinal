from flask import Flask, request, jsonify
import cv2, numpy as np, joblib, os
from keras.models import load_model

# สร้างแอพ Flask
app = Flask(__name__)

# --- พาธไฟล์โมเดลและ labels ---
# โมเดลและไฟล์ labels ควรถูกเซฟไว้ที่โฟลเดอร์ `models/`
MODEL_PATH = os.path.join("models", "chars74k_model.h5")
LABEL_PATH = os.path.join("models", "chars74k_labels.pkl")


# --- โหลดโมเดลและ mapping รายการ label ---
# - โมเดล Keras (h5) สำหรับจำแนกตัวอักษร
# - label_list เก็บลำดับคลาสที่สอดคล้องกับ output ของ softmax
model = load_model(MODEL_PATH)
label_list = joblib.load(LABEL_PATH)  # เช่น ['0','1',...'A',...]


# ขนาดภาพที่โมเดลคาดหวัง (ต้องตรงกับตัวที่ฝึกมา)
IMG_SIZE = 32


def preprocess(image):
    """
    เตรียมภาพก่อนป้อนเข้าโมเดล:
    - ย่อ/ขยายเป็น (IMG_SIZE, IMG_SIZE)
    - แปลง BGR -> RGB (OpenCV โหลดเป็น BGR แต่ training ใช้ RGB)
    - แปลงเป็น float และ normalize เป็น 0..1
    - เพิ่มมิติ batch (1, H, W, C)
    """
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint รับไฟล์รูปภาพจาก client (form-data field ชื่อ 'file')
    - อ่านภาพจาก request.files
    - ตรวจสอบความถูกต้อง และเรียก preprocess -> model.predict
    - ส่งกลับ JSON ที่มี `label` และ `confidence`
    """
    # ตรวจว่ามีไฟล์ใน request หรือไม่
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]

    # อ่านไบต์แล้ว decode เป็นภาพ OpenCV
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        # คืนค่า error ถ้า decode ไม่สำเร็จ
        return jsonify({"error": "Invalid image"}), 400

    # เตรียมภาพและทำนาย
    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]

    # หา index ของคลาสที่มีความน่าจะเป็นสูงสุด
    idx = int(np.argmax(probs))
    label = label_list[idx]
    conf = float(probs[idx])

    # ส่งผลลัพธ์กลับเป็น JSON
    return jsonify({"label": label, "confidence": conf})


if __name__ == "__main__":
    # เริ่มเซิร์ฟเวอร์ Flask บนพอร์ต 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
