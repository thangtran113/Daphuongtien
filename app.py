import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from PIL import Image

# Hàm trích xuất đặc trưng trung bình
def get_feature_vector(img_path, algo="ORB"):
    if algo == "SIFT":
        detector = cv2.SIFT_create()
    elif algo == "ORB":
        detector = cv2.ORB_create()
    else:
        raise ValueError("Algorithm must be 'SIFT' or 'ORB'.")

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, descriptors = detector.detectAndCompute(gray, None)

    return np.mean(descriptors, axis=0) if descriptors is not None else None

# Hàm lưu đặc trưng vào file CSV
def save_features_to_csv(img_dir, algo, output_file):
    feature_data = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        try:
            feature_vector = get_feature_vector(img_path, algo)
            if feature_vector is not None:
                feature_data.append({
                    "image_name": img_name,
                    "features": feature_vector.tolist()
                })
        except Exception as e:
            st.warning(f"Không thể xử lý ảnh {img_name}: {e}")
    
    # Lưu ra file CSV
    df = pd.DataFrame(feature_data)
    df.to_csv(output_file, index=False)
    st.success(f"Lưu đặc trưng vào tệp: {output_file}")

    # Hiển thị bảng đặc trưng
    st.write("### Đặc trưng của từng ảnh:")
    st.dataframe(df)

# Hàm tìm kiếm ảnh tương tự
def find_similar_images(query_vector, csv_file, top_k=5):
    df = pd.read_csv(csv_file)
    df["features"] = df["features"].apply(eval)  # Chuyển chuỗi về mảng
    distances = []
    for _, row in df.iterrows():
        dist = np.linalg.norm(query_vector - np.array(row["features"]))
        distances.append((row["image_name"], dist, row["features"]))
    distances = sorted(distances, key=lambda x: x[1])
    return distances[:top_k]

# Hàm hiển thị kết quả
def display_results(query_img_path, similar_images, img_dir):
    st.image(query_img_path, caption="Ảnh truy vấn", use_column_width=True)
    st.write("### Kết quả tìm kiếm:")
    cols = st.columns(len(similar_images))
    for i, (img_name, dist, features) in enumerate(similar_images):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        cols[i].image(img, caption=f"{img_name}\nDistance: {dist:.2f}")

    # Hiển thị đặc trưng chi tiết
    st.write("### Đặc trưng của ảnh tương tự:")
    feature_df = pd.DataFrame(similar_images, columns=["image_name", "distance", "features"])
    st.dataframe(feature_df[["image_name", "distance"]])  # Hiển thị tên ảnh và khoảng cách
    st.json(feature_df.to_dict(orient="records"))         # Hiển thị đặc trưng chi tiết dưới dạng JSON

# Giao diện Streamlit
st.title("Tìm kiếm ảnh tương tự")
st.write("Ứng dụng tìm kiếm ảnh tương tự dựa trên đặc trưng SIFT hoặc ORB.")

# Chọn thư mục ảnh
img_dir = st.text_input("Thư mục chứa ảnh", "test_img")
algo = st.selectbox("Chọn thuật toán trích xuất đặc trưng", ["ORB", "SIFT"])

# Đặt tên tệp CSV theo thuật toán
csv_file = f"features_{algo.lower()}.csv"

# Lưu đặc trưng
if st.button(f"Lưu đặc trưng {algo} vào CSV"):
    if os.path.exists(img_dir):
        save_features_to_csv(img_dir, algo, csv_file)
    else:
        st.error("Thư mục ảnh không tồn tại!")

# Upload ảnh truy vấn
uploaded_file = st.file_uploader("Chọn ảnh truy vấn", type=["jpg", "jpeg", "png"])
find_button = st.button("Tìm kiếm ảnh tương tự")

if uploaded_file and find_button:
    # Lưu ảnh truy vấn tạm thời
    query_img_path = f"temp_{uploaded_file.name}"
    with open(query_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Tính vector đặc trưng của ảnh truy vấn
        query_vector = get_feature_vector(query_img_path, algo)

        # Tìm kiếm ảnh tương tự
        if not os.path.exists(csv_file):
            st.error(f"Không tìm thấy file đặc trưng: {csv_file}. Vui lòng lưu đặc trưng trước!")
        else:
            similar_images = find_similar_images(query_vector, csv_file, top_k=5)

            # Hiển thị kết quả
            display_results(query_img_path, similar_images, img_dir)

    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
