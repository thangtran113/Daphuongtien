# Import các thư viện cần thiết
import cv2
import os
import numpy as np
import pickle
# Hàm trích xuất đặc trưng từ ảnh
def extract_features(img_path, output_path, algo="SIFT", max_keypoints=500):
    """
    Trích xuất đặc trưng từ ảnh và lưu kết quả vào tệp.
    :param img_path: Đường dẫn thư mục chứa ảnh.
    :param output_path: Đường dẫn lưu trữ tệp đặc trưng.
    :param algo: Thuật toán sử dụng (SIFT hoặc ORB).
    :param max_keypoints: Số lượng điểm đặc trưng tối đa.
    """
    # Lấy danh sách các ảnh
    img_list = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Khởi tạo thuật toán
    if algo == "SIFT":
        detector = cv2.SIFT_create(max_keypoints)
    elif algo == "ORB":
        detector = cv2.ORB_create(max_keypoints)
    else:
        raise ValueError("Algorithm must be 'SIFT' or 'ORB'.")

    # Lưu trữ đặc trưng
    features = {}

    for img_name in img_list:
        img_file = os.path.join(img_path, img_name)
        print(f"Đang xử lý: {img_file}")

        # Đọc và chuyển đổi ảnh
        img = cv2.imread(img_file)
        if img is None:
            print(f"Không thể đọc ảnh: {img_file}. Bỏ qua...")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Trích xuất đặc trưng
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        features[img_name] = {
            "keypoints": [kp.pt for kp in keypoints],  # Chỉ lưu tọa độ điểm đặc trưng
            "descriptors": descriptors
        }

    print(features)
    # Lưu kết quả vào tệp
    with open(output_path, "wb") as f:
        pickle.dump(features, f)
    
    print(f"Đặc trưng đã được lưu vào: {output_path}")

# Đường dẫn đến thư mục chứa ảnh
img_path = "test_img/"

# Đường dẫn lưu đặc trưng
output_path = "features.pkl"

# Gọi hàm trích xuất
extract_features(img_path, output_path, max_keypoints=500)
