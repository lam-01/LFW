import streamlit as st
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Updated load_data function
@st.cache_data
def load_data(sample_size=None):
    lfw = fetch_lfw_people(resize=0.4, color=False)
    X, y = lfw.data, lfw.target
    target_names = lfw.target_names
    X = X / 255.0

    unique_labels, counts = np.unique(y, return_counts=True)
    min_images_per_person = min(counts)
    
    balanced_X = []
    balanced_y = []
    for label in unique_labels:
        indices = np.where(y == label)[0]
        sampled_indices = np.random.choice(indices, min_images_per_person, replace=False)
        balanced_X.append(X[sampled_indices])
        balanced_y.append(y[sampled_indices])
    
    X = np.vstack(balanced_X)
    y = np.hstack(balanced_y)

    if sample_size is not None and sample_size < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=sample_size, random_state=42)

    return X, y, target_names

# [split_data, train_model, preprocess_uploaded_image, show_sample_images remain unchanged]

def create_streamlit_app():
    st.title("👤 Nhận diện khuôn mặt với LFW")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📓 Lí thuyết", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])
    
    with tab1:
        # [Unchanged]
        algorithm = st.selectbox("Chọn thuật toán:", ["SVM", "CNN"])
        if algorithm == "SVM":
            st.write("##### Support Vector Machine (SVM)")
            st.write("###### Các kernel trong SVM")
            st.write("**1. Linear Kernel**")
            st.latex(r"K(x, x') = x \cdot x'")
            st.write("**2. RBF Kernel**")
            st.latex(r"K(x, x') = \exp\left(-\frac{||x - x'||^2}{2\sigma^2}\right)")
            st.write("**3. Polynomial Kernel**")
            st.latex(r"K(x, x') = (x \cdot x' + c)^d")
            st.write("**4. Sigmoid Kernel**")
            st.latex(r"K(x, x') = \tanh(\alpha \cdot (x \cdot x') + c)")
        elif algorithm == "CNN":
            st.write("##### Convolutional Neural Network (CNN)")
            st.write("- **Convolutional Layers**: Trích xuất đặc trưng không gian từ ảnh.")
            st.write("- **Pooling Layers**: Giảm kích thước không gian, giữ lại thông tin quan trọng.")
            st.write("- **Fully Connected Layers**: Phân loại dựa trên đặc trưng đã trích xuất.")
            st.latex(r"y = \text{softmax}(W \cdot x + b)")

    with tab2:
        sample_size = st.number_input("Cỡ mẫu huấn luyện", 100, 5000, 1000, step=100)
        X, y, target_names = load_data(sample_size=sample_size)
        img_shape = (50, 37)
        st.write(f"**Số lượng mẫu: {X.shape[0]}, Số người: {len(target_names)}**")
        st.write(f"**Số ảnh mỗi người: {X.shape[0] // len(target_names)}**")
        show_sample_images(X, y, target_names, img_shape)

        test_size = st.slider("Tỷ lệ Test (%)", 5, 30, 15, step=5)
        val_size = st.slider("Tỷ lệ Validation (%)", 5, 30, 15, step=5)
        train_size = 100 - test_size
        val_ratio = val_size / train_size

        if val_ratio >= 1.0:
            st.error("Tỷ lệ Validation quá lớn!")
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=test_size/100, val_size=val_size/100)
            data_ratios = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Validation", "Test"],
                "Tỷ lệ (%)": [train_size - val_size, val_size, test_size],
                "Số lượng mẫu": [len(X_train), len(X_val), len(X_test)]
            })
            st.table(data_ratios)

        st.write("**🚀 Huấn luyện mô hình**")
        custom_model_name = st.text_input("Nhập tên mô hình:", "Default_model")
        model_name = st.selectbox("🔍 Chọn mô hình", ["SVM", "CNN"])
        params = {}

        if model_name == "SVM":
            params["kernel"] = st.selectbox("⚙️ Kernel", ["linear", "rbf", "poly", "sigmoid"])
            params["C"] = st.slider("🔧 Tham số C", 0.1, 10.0, 1.0)
        elif model_name == "CNN":
            params["epochs"] = st.slider("🔄 Số epoch", 5, 50, 10)

        if st.button("🚀 Huấn luyện"):
            with st.spinner("🔄 Đang huấn luyện..."):
                model, train_acc, val_acc, test_acc = train_model(
                    custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, img_shape
                )
            if model is not None:
                st.success("✅ Huấn luyện xong!")
                st.write(f"🎯 Train Accuracy: {train_acc:.4f}")
                st.write(f"🎯 Validation Accuracy: {val_acc:.4f}")
                st.write(f"🎯 Test Accuracy: {test_acc:.4f}")

    with tab3:
        # [Unchanged from previous update]
        st.write("##### 🔮 Dự đoán trên ảnh tải lên")
        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            runs["model_custom_name"] = runs["tags.mlflow.runName"]
            available_models = runs["model_custom_name"].dropna().unique().tolist()
        else:
            available_models = []

        if available_models:
            selected_model_name = st.selectbox("📝 Chọn mô hình đã huấn luyện:", available_models)
            selected_run = runs[runs["model_custom_name"] == selected_model_name].iloc[0]
            run_id = selected_run["run_id"]
            model_type = selected_run["params.model_name"]
            model_uri = f"runs:/{run_id}/{model_type}"
            try:
                if model_type == "SVM":
                    model = mlflow.sklearn.load_model(model_uri)
                elif model_type == "CNN":
                    model = mlflow.tensorflow.load_model(model_uri)
                st.success(f"✅ Đã tải mô hình: `{selected_model_name}` (Loại: {model_type})")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
                model = None
        else:
            st.warning("⚠️ Không có mô hình nào được lưu trong MLflow.")
            model = None

        img_shape = (50, 37)
        uploaded_file = st.file_uploader("📤 Tải ảnh khuôn mặt (PNG, JPG)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None and model is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            processed_image = preprocess_uploaded_image(image, img_shape)
            st.image(image, caption="📷 Ảnh tải lên", use_column_width=True)
            
            if st.button("🔮 Dự đoán"):
                try:
                    if model_type == "SVM":
                        pred = model.predict(processed_image)[0]
                        probs = model.predict_proba(processed_image)[0]
                    elif model_type == "CNN":
                        processed_image_reshaped = processed_image.reshape((1, *img_shape, 1))
                        pred = np.argmax(model.predict(processed_image_reshaped), axis=1)[0]
                        probs = model.predict(processed_image_reshaped)[0]
                    
                    st.write(f"🎯 **Dự đoán: {target_names[pred]}**")
                    st.write(f"🔢 **Độ tin cậy: {probs[pred] * 100:.2f}%**")
                except Exception as e:
                    st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
        elif uploaded_file is not None and model is None:
            st.error("❌ Vui lòng chọn một mô hình hợp lệ trước khi dự đoán.")

    with tab4:
        # [Unchanged]
        st.write("##### 📊 MLflow Tracking")
        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            runs["model_custom_name"] = runs["tags.mlflow.runName"]
            search_model_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")
            filtered_runs = runs[runs["model_custom_name"].str.contains(search_model_name, case=False, na=False)] if search_model_name else runs
            if not filtered_runs.empty:
                st.dataframe(filtered_runs[["model_custom_name", "params.model_name", "metrics.train_accuracy", 
                                           "metrics.val_accuracy", "metrics.test_accuracy"]])
                selected_model = st.selectbox("📝 Chọn mô hình để xem chi tiết:", filtered_runs["model_custom_name"].tolist())
                run_details = mlflow.get_run(filtered_runs[filtered_runs["model_custom_name"] == selected_model].iloc[0]["run_id"])
                st.write(f"##### 🔍 Chi tiết mô hình: `{selected_model}`")
                st.write("📌 **Tham số:**", run_details.data.params)
                st.write("📊 **Metric:**", run_details.data.metrics)
            else:
                st.write("❌ Không tìm thấy mô hình.")
        else:
            st.write("⚠️ Không có phiên làm việc nào được ghi lại.")

if __name__ == "__main__":
    create_streamlit_app()
