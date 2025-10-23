import cv2
import os
import numpy as np
from flask import Flask, request, jsonify # Dùng Flask để tạo API
import io

# === 1. TẢI CÁC MÔ HÌNH (Logic từ file recognize.py của bạn) ===

# Tải bộ phát hiện khuôn mặt (haarcascade)
try:
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"[LỖI] Không thể tải 'haarcascade_frontalface_default.xml'. Đảm bảo file tồn tại. Lỗi: {e}")
    exit()

# Tải bộ nhận diện (LBPH)
clf = cv2.face.LBPHFaceRecognizer_create()

# LƯU Ý: File classifier.py của bạn tạo ra "classifier.xml"
# File recognize.py cuối cùng của bạn cũng đọc "classifier.xml"
MODEL_FILE = "classifier.xml" 

if not os.path.exists(MODEL_FILE):
    print(f"[LỖI] Không tìm thấy file mô hình '{MODEL_FILE}'")
    print("Vui lòng chạy file classifier.py ở local và upload file .xml lên.")
    exit()

try:
    clf.read(MODEL_FILE)
    print(f"[INFO] Đã tải mô hình đã huấn luyện từ '{MODEL_FILE}'")
except cv2.error as e:
    print(f"[LỖI] OpenCV không thể đọc file '{MODEL_FILE}'. File có thể bị hỏng. Lỗi: {e}")
    exit()

# === 2. ĐỊNH NGHĨA NGƯỠNG VÀ TÊN ===
# Đặt ngưỡng tin cậy. Với LBPH, 0 là hoàn hảo, 100 là khá khác.
# Bất cứ ai có độ tin cậy > 100 chắc chắn là người lạ.
THRESHOLD = 100 
RECOGNIZED_ID = 1 # ID của người bạn muốn nhận diện
RECOGNIZED_NAME = "BAO" # Tên của ID 1, theo file recognize.py của bạn

# === 3. KHỞI TẠO FLASK APP ===
app = Flask(__name__)

# === 4. HÀM XỬ LÝ NHẬN DIỆN (Lõi logic từ recognize.py) ===
# Hàm này lấy logic từ hàm draw_boundary trong recognize.py
def process_image(img_bytes):
    try:
        # 1. Giải mã (decode) ảnh từ bytes mà ESP32 gửi lên
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "sai", "reason": "Khong the decode anh"}

        # 2. Chuyển sang ảnh xám
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Phát hiện khuôn mặt
        features = faceCascade.detectMultiScale(gray_img, 1.1, 10)

        if len(features) == 0:
            # Không tìm thấy mặt
            return {"status": "sai", "reason": "Khong tim thay mat"}

        # 4. Chỉ xử lý khuôn mặt đầu tiên tìm thấy
        (x, y, w, h) = features[0]
        roi_gray = gray_img[y:y+h, x:x+w]
        
        # 5. Dự đoán (giống recognize.py)
        id, confidence = clf.predict(roi_gray)
        
        # 6. KIỂM TRA: Nếu ID đúng VÀ độ tin cậy đủ tốt
        if id == RECOGNIZED_ID and confidence < THRESHOLD:
            print(f"Phat hien: {RECOGNIZED_NAME} (Confidence: {confidence})")
            # Trả về "dung"
            return {"status": "dung", "name": RECOGNIZED_NAME, "confidence": round(confidence, 2)}
        else:
            # Là người lạ (hoặc nhận diện sai)
            print(f"Phat hien: Nguoi la (ID: {id}, Confidence: {confidence})")
            # Trả về "sai"
            return {"status": "sai", "reason": "Nguoi la"}

    except Exception as e:
        print(f"Loi xu ly anh: {e}")
        return {"status": "sai", "reason": str(e)}

# === 5. TẠO API ENDPOINT CHO ESP32 ===
@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    # 1. Kiểm tra xem có file 'image' được gửi lên không
    if 'image' not in request.files:
        return jsonify({"status": "sai", "reason": "Khong co file 'image'"})
    
    file = request.files['image']
    
    # 2. Đọc dữ liệu bytes của file
    img_bytes = file.read()
    
    # 3. Gọi hàm xử lý ảnh
    result = process_image(img_bytes)
    
    # 4. Trả kết quả JSON về cho ESP32
    return jsonify(result)

# Route cơ bản để kiểm tra server có chạy không
@app.route('/', methods=['GET'])
def index():
    return "Face Recognition API Server is running. (OK)"

# === 6. CHẠY SERVER ===
if __name__ == "__main__":
    # Khi deploy lên Render, Gunicorn sẽ chạy file này
    # Dòng này chỉ dùng để bạn test ở máy local
    app.run(debug=True, host='0.0.0.0', port=5000)