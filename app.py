import streamlit as st
import cv2
import os
import numpy as np
import pickle
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

# Hàm tìm kiếm ảnh tương tự
def find_similar_images(query_vector, features_file, top_k=5):
    with open(features_file, "rb") as f:
        features = pickle.load(f)

    distances = []
    for img_name, data in features.items():
        feature_vector = np.mean(data["descriptors"], axis=0) if data["descriptors"] is not None else None
        if feature_vector is not None:
            dist = np.linalg.norm(query_vector - feature_vector)
            distances.append((img_name, dist))
    distances = sorted(distances, key=lambda x: x[1])
    return distances[:top_k]

# Hàm hiển thị kết quả
def display_results(query_img_path, similar_images, img_dir):
    st.image(query_img_path, caption="Ảnh truy vấn", use_column_width=True)
    st.write("### Kết quả tìm kiếm:")
    cols = st.columns(len(similar_images))
    for i, (img_name, dist) in enumerate(similar_images):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        cols[i].image(img, caption=f"{img_name}\nDistance: {dist:.2f}")

# Giao diện Streamlit
st.title("Tìm kiếm ảnh tương tự")
st.write("Ứng dụng tìm kiếm ảnh tương tự dựa trên đặc trưng SIFT hoặc ORB.")

# Upload ảnh truy vấn
uploaded_file = st.file_uploader("Chọn ảnh truy vấn", type=["jpg", "jpeg", "png"])
algo = st.selectbox("Chọn thuật toán trích xuất đặc trưng", ["ORB", "SIFT"])
num_clusters = st.slider("Số cụm K-Means", min_value=2, max_value=10, value=5)
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
        features_file = "features.pkl"  # Tệp đặc trưng
        img_dir = "test_img"  # Thư mục ảnh
        similar_images = find_similar_images(query_vector, features_file, top_k=5)

        # Hiển thị kết quả
        display_results(query_img_path, similar_images, img_dir)

    except Exception as e:
        st.error(f"Lỗi: {str(e)}")

# Phân cụm
if st.button("Phân cụm K-Means"):
    try:
        with open("features.pkl", "rb") as f:
            features = pickle.load(f)

        feature_vectors = [
            np.mean(data["descriptors"], axis=0)
            for data in features.values()
            if data["descriptors"] is not None
        ]
        img_names = [name for name, data in features.items() if data["descriptors"] is not None]

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feature_vectors)
        st.write("Phân cụm hoàn tất!")
        cluster_labels = kmeans.labels_

        # Hiển thị các cụm
        st.write("### Ảnh trong từng cụm:")
        for cluster_id in range(num_clusters):
            st.write(f"#### Cụm {cluster_id + 1}:")
            cols = st.columns(5)
            cluster_images = [img_names[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            for i, img_name in enumerate(cluster_images):
                img_path = os.path.join("test_img", img_name)
                img = Image.open(img_path)
                cols[i % 5].image(img, caption=img_name, use_column_width=True)

    except Exception as e:
        st.error(f"Lỗi phân cụm: {str(e)}")
