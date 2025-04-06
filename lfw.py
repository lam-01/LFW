import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# Khởi tạo session state
if 'model_svm' not in st.session_state:
    st.session_state.model_svm = None
if 'model_nn' not in st.session_state:
    st.session_state.model_nn = None
if 'data_split' not in st.session_state:
    st.session_state.data_split = None
if 'params_svm' not in st.session_state:
    st.session_state.params_svm = None
if 'params_nn' not in st.session_state:
    st.session_state.params_nn = None
if 'cv_folds' not in st.session_state:
    st.session_state.cv_folds = 3
if 'custom_model_name' not in st.session_state:
    st.session_state.custom_model_name = ""
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# Đường dẫn cố định đến thư mục dataset
DATASET_PATH = "G:\Download\animals"

# Tải dữ liệu từ thư mục
@st.cache_data
def load_data(dataset_path=DATASET_PATH, n_samples=None):
    cat_path = os.path.join(dataset_path, "cats")
    dog_path = os.path.join(dataset_path, "dogs")
    
    if not os.path.exists(cat_path) or not os.path.exists(dog_path):
        st.error(f"Không tìm thấy thư mục: {cat_path} hoặc {dog_path}. Vui lòng kiểm tra đường dẫn!")
        return None, None
    
    X = []
    y = []
    
    # Tải ảnh mèo (nhãn 0)
    for img_file in os.listdir(cat_path):
        img_path = os.path.join(cat_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            st.warning(f"Không thể đọc file: {img_path}")
            continue
        img = cv2.resize(img, (64, 64))  # Giảm kích thước để xử lý nhanh
        X.append(img.flatten())
        y.append(0)
    
    # Tải ảnh chó (nhãn 1)
    for img_file in os.listdir(dog_path):
        img_path = os.path.join(dog_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            st.warning(f"Không thể đọc file: {img_path}")
            continue
        img = cv2.resize(img, (64, 64))
        X.append(img.flatten())
        y.append(1)
    
    if not X:
        st.error("Không có ảnh nào được tải. Kiểm tra lại thư mục hoặc định dạng file!")
        return None, None
    
    X = np.array(X) / 255.0  # Chuẩn hóa
    y = np.array(y)
    
    if n_samples:
        X, y = X[:n_samples], y[:n_samples]
    
    return X, y

# Chia dữ liệu
@st.cache_data
def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (train_size + val_size), random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Visualize mạng nơ-ron
def visualize_neural_network_prediction(model, input_image, predicted_label):
    hidden_layer_sizes = model.hidden_layer_sizes
    if isinstance(hidden_layer_sizes, int):
        hidden_layer_sizes = [hidden_layer_sizes]
    elif isinstance(hidden_layer_sizes, tuple):
        hidden_layer_sizes = list(hidden_layer_sizes)

    input_layer_size = 64*64
    output_layer_size = 2
    layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
    num_layers = len(layer_sizes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 3]})
    ax1.imshow(input_image.reshape(64, 64), cmap='gray')
    ax1.set_title("Input Image")
    ax1.axis('off')

    pos = {}
    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(hidden_layer_sizes))] + ['Output']

    for layer_idx, layer_size in enumerate(layer_sizes):
        for neuron_idx in range(layer_size):
            if layer_size > 20 and layer_idx == 0:
                if neuron_idx < 10 or neuron_idx >= layer_size - 10:
                    pos[(layer_idx, neuron_idx)] = (layer_idx, neuron_idx / layer_size)
                elif neuron_idx == 10:
                    pos[('dots', layer_idx)] = (layer_idx, 0.5)
            else:
                pos[(layer_idx, neuron_idx)] = (layer_idx, neuron_idx / (layer_size - 1) if layer_size > 1 else 0.5)

    for layer_idx, layer_size in enumerate(layer_sizes):
        for neuron_idx in range(layer_size):
            if layer_size > 20 and layer_idx == 0 and neuron_idx >= 10 and neuron_idx < layer_size - 10:
                continue
            x, y = pos[(layer_idx, neuron_idx)]
            circle = Circle((x, y), 0.05, color='white', ec='black')
            ax2.add_patch(circle)
            if layer_idx == num_layers - 1:
                ax2.text(x + 0.2, y, f"{'Mèo' if neuron_idx == 0 else 'Chó'}", fontsize=12, color='white')
            if layer_idx == num_layers - 1 and neuron_idx == predicted_label:
                square = Rectangle((x - 0.07, y - 0.07), 0.14, 0.14, fill=False, edgecolor='yellow', linewidth=2)
                ax2.add_patch(square)

    if ('dots', 0) in pos:
        x, y = pos[('dots', 0)]
        ax2.text(x, y, "...", fontsize=12, color='white', ha='center', va='center')

    for layer_idx in range(len(layer_sizes) - 1):
        current_layer_size = layer_sizes[layer_idx]
        next_layer_size = layer_sizes[layer_idx + 1]
        neuron_indices_1 = range(min(10, current_layer_size)) if current_layer_size > 20 else range(current_layer_size)
        neuron_indices_2 = range(next_layer_size)

        for idx1, neuron1 in enumerate(neuron_indices_1):
            for idx2, neuron2 in enumerate(neuron_indices_2):
                x1, y1 = pos[(layer_idx, neuron1)]
                x2, y2 = pos[(layer_idx + 1, neuron2)]
                color = plt.cm.coolwarm(idx2 / max(len(neuron_indices_2), 1))
                ax2.plot([x1, x2], [y1, y2], color=color, alpha=0.5, linewidth=1)

    ax2.set_xlim(-0.5, num_layers - 0.5)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xticks(range(num_layers))
    ax2.set_xticklabels(layer_names)
    ax2.set_yticks([])
    ax2.set_title(f"Dự đoán: {'Mèo' if predicted_label == 0 else 'Chó'}")
    ax2.set_facecolor('black')
    return fig

# Huấn luyện SVM
@st.cache_resource
def train_svm(custom_model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds):
    model = SVC(kernel=params["kernel"], C=params["C"], random_state=42)
    with mlflow.start_run(run_name=custom_model_name):
        model.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        cv_scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42))
        cv_mean_accuracy = np.mean(cv_scores)

        mlflow.log_param("model_name", "SVM")
        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("cv_mean_accuracy", cv_mean_accuracy)
        mlflow.sklearn.log_model(model, "SVM")
    return model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy

# Huấn luyện Neural Network
@st.cache_resource
def train_nn(custom_model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds):
    hidden_layer_sizes = tuple([params["neurons_per_layer"]] * params["num_hidden_layers"])
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=params["epochs"],
        activation=params["activation"],
        learning_rate_init=params["learning_rate"],
        solver='adam',
        random_state=42
    )
    with mlflow.start_run(run_name=custom_model_name):
        model.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        cv_scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42))
        cv_mean_accuracy = np.mean(cv_scores)

        mlflow.log_param("model_name", "Neural Network")
        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("cv_mean_accuracy", cv_mean_accuracy)
        mlflow.sklearn.log_model(model, "Neural Network")
    return model, train_accuracy, val_accuracy, test_accuracy, cv_mean_accuracy

# Xử lý ảnh tải lên
def preprocess_uploaded_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return image.reshape(1, -1)

# Xử lý ảnh từ canvas
def preprocess_canvas_image(canvas):
    image = np.array(canvas)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    image = cv2.bitwise_not(image)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return image.reshape(1, -1)

# Hiển thị mẫu dữ liệu
def show_sample_images(X, y):
    st.write("**🖼️ Một vài mẫu dữ liệu**")
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i, ax in enumerate(axes):
        idx = np.where(y == i % 2)[0][i // 2]
        ax.imshow(X[idx].reshape(64, 64), cmap='gray')
        ax.set_title(f"{'Mèo' if i % 2 == 0 else 'Chó'}")
        ax.axis('off')
    st.pyplot(fig)

# Giao diện Streamlit
def create_streamlit_app():
    st.title("🐱🐶 Phân loại ảnh mèo/chó")

    tab1, tab2, tab3, tab4 = st.tabs(["📓 Lý thuyết", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])

    with tab1:
        st.write("##### SVM và Neural Network")
        st.write("SVM (Support Vector Machine) và Neural Network là hai phương pháp học máy phổ biến để phân loại dữ liệu, chẳng hạn như phân biệt ảnh mèo và chó.")
        st.write("**SVM**: Tìm siêu phẳng tối ưu để phân tách các lớp dữ liệu. Phù hợp với dữ liệu tuyến tính và phi tuyến tính (với kernel).")
        st.write("**Neural Network**: Mô phỏng mạng nơ-ron sinh học, học các đặc trưng phức tạp qua các tầng ẩn.")
        st.write("Tập dữ liệu: 1000 ảnh (500 mèo, 500 chó), độ phân giải 512x512, định dạng .png, được tạo bởi Stable Diffusion 1.5.")

    with tab2:
        n_samples = st.number_input("Số lượng mẫu", min_value=100, max_value=1000, value=500, step=50)
        X, y = load_data(n_samples=n_samples)
        
        if X is None or y is None:
            st.error("Không thể tải dữ liệu. Vui lòng kiểm tra thư mục 'dataset/'!")
        else:
            st.write(f"**Số lượng mẫu được chọn: {X.shape[0]}**")
            show_sample_images(X, y)

            test_size = st.slider("Tỷ lệ Test (%)", min_value=5, max_value=30, value=15, step=5)
            val_size = st.slider("Tỷ lệ Validation (%)", min_value=5, max_value=30, value=15, step=5)
            train_size = 100 - test_size - val_size
            if train_size <= 0:
                st.error("Tỷ lệ không hợp lệ! Tổng Train + Val + Test phải = 100%.")
            else:
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_size=train_size/100, val_size=val_size/100, test_size=test_size/100)
                st.session_state.data_split = (X_train, X_val, X_test, y_train, y_val, y_test)

                st.write("**🚀 Huấn luyện mô hình**")
                model_type = st.selectbox("Chọn loại mô hình", ["SVM", "Neural Network"])
                st.session_state.custom_model_name = st.text_input("Tên mô hình để lưu vào MLflow:")

                if model_type == "SVM":
                    params = {
                        "kernel": st.selectbox("Kernel", ["linear", "rbf"]),
                        "C": st.slider("C (Regularization)", 0.1, 10.0, 1.0)
                    }
                    st.session_state.params_svm = params
                else:
                    params = {
                        "num_hidden_layers": st.slider("Số lớp ẩn", 1, 2, 1),
                        "neurons_per_layer": st.slider("Số neuron mỗi lớp", 20, 100, 50),
                        "epochs": st.slider("Epochs", 5, 50, 10),
                        "activation": st.selectbox("Hàm kích hoạt", ["relu", "tanh", "logistic"]),
                        "learning_rate": st.slider("Tốc độ học", 0.0001, 0.1, 0.001)
                    }
                    st.session_state.params_nn = params

                if st.button("🚀 Huấn luyện"):
                    if not st.session_state.custom_model_name:
                        st.error("Vui lòng nhập tên mô hình!")
                    else:
                        X_train, X_val, X_test, y_train, y_val, y_test = st.session_state.data_split
                        if model_type == "SVM":
                            result = train_svm(st.session_state.custom_model_name, st.session_state.params_svm, X_train, X_val, X_test, y_train, y_val, y_test, st.session_state.cv_folds)
                            st.session_state.model_svm = result[0]
                        else:
                            result = train_nn(st.session_state.custom_model_name, st.session_state.params_nn, X_train, X_val, X_test, y_train, y_val, y_test, st.session_state.cv_folds)
                            st.session_state.model_nn = result[0]
                        st.session_state.trained_models[st.session_state.custom_model_name] = result[0]
                        st.success("✅ Huấn luyện xong!")
                        st.write(f"Độ chính xác Train: {result[1]:.4f}")
                        st.write(f"Độ chính xác Validation: {result[2]:.4f}")
                        st.write(f"Độ chính xác Test: {result[3]:.4f}")
                        st.write(f"Độ chính xác CV: {result[4]:.4f}")

    with tab3:
        if not st.session_state.trained_models:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước!")
        else:
            model_names = list(st.session_state.trained_models.keys())
            selected_model_name = st.selectbox("Chọn mô hình để dự đoán:", model_names)
            selected_model = st.session_state.trained_models[selected_model_name]

            option = st.radio("Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ ảnh"])
            show_visualization = st.checkbox("Hiển thị biểu đồ mạng nơ-ron (chỉ cho NN)", value=True)

            if option == "📂 Tải ảnh lên":
                uploaded_file = st.file_uploader("Tải ảnh mèo/chó", type=["png"])
                if uploaded_file:
                    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                    processed_image = preprocess_uploaded_image(image)
                    st.image(image, caption="Ảnh tải lên", use_column_width=True)
                    if st.button("🔮 Dự đoán"):
                        prediction = selected_model.predict(processed_image)[0]
                        st.write(f"🎯 Dự đoán: {'Mèo' if prediction == 0 else 'Chó'}")
                        if isinstance(selected_model, MLPClassifier) and show_visualization:
                            fig = visualize_neural_network_prediction(selected_model, processed_image, prediction)
                            st.pyplot(fig)

            elif option == "✏️ Vẽ ảnh":
                canvas_result = st_canvas(
                    fill_color="white", stroke_width=15, stroke_color="black",
                    background_color="white", width=280, height=280, drawing_mode="freedraw", key="canvas"
                )
                if st.button("🔮 Dự đoán"):
                    if canvas_result.image_data is not None:
                        processed_canvas = preprocess_canvas_image(canvas_result.image_data)
                        prediction = selected_model.predict(processed_canvas)[0]
                        st.write(f"🎯 Dự đoán: {'Mèo' if prediction == 0 else 'Chó'}")
                        if isinstance(selected_model, MLPClassifier) and show_visualization:
                            fig = visualize_neural_network_prediction(selected_model, processed_canvas, prediction)
                            st.pyplot(fig)

    with tab4:
        st.write("##### 📊 MLflow Tracking")
        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            if "tags.mlflow.runName" in runs.columns:
                runs["model_custom_name"] = runs["tags.mlflow.runName"]
            else:
                runs["model_custom_name"] = "Unnamed Model"
            st.dataframe(runs[["model_custom_name", "params.model_name", "metrics.train_accuracy", "metrics.test_accuracy"]])
        else:
            st.write("⚠️ Không có phiên làm việc nào được ghi lại.")

if __name__ == "__main__":
    create_streamlit_app()
