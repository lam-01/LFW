import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import os
from glob import glob

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

# Tải dữ liệu mèo/chó từ thư mục
@st.cache_data
def load_data(data_dir, n_samples=None):
    cat_images = glob(os.path.join(data_dir, "cats", "*.jpg")) + glob(os.path.join(data_dir, "cats", "*.png"))
    dog_images = glob(os.path.join(data_dir, "dogs", "*.jpg")) + glob(os.path.join(data_dir, "dogs", "*.png"))
    images = cat_images + dog_images
    labels = [0] * len(cat_images) + [1] * len(dog_images)
    
    X, y = [], []
    for img_path, label in zip(images[:n_samples], labels[:n_samples]):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)) / 255.0
            X.append(img.flatten())
            y.append(label)
    return np.array(X), np.array(y)

# Chia dữ liệu thành train, validation, test
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

    input_layer_size = 64 * 64  # Ảnh 64x64
    output_layer_size = 2  # Mèo hoặc chó
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
    model = SVC(kernel=params["kernel"], C=params["C"], probability=True, random_state=42)
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
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        idx = np.where(y == i % 2)[0][i // 2]
        ax.imshow(X[idx].reshape(64, 64), cmap='gray')
        ax.set_title(f"{'Mèo' if y[idx] == 0 else 'Chó'}")
        ax.axis('off')
    st.pyplot(fig)

# Giao diện Streamlit
def create_streamlit_app():
    st.title("🐱🐶 Phân loại ảnh mèo/chó")

    tab1, tab2, tab3, tab4 = st.tabs(["📓 Lý thuyết", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])

    with tab1:
        st.write("##### SVM và Neural Network")
        st.write("SVM (Support Vector Machine) và Neural Network là hai phương pháp học máy phổ biến để phân loại dữ liệu, chẳng hạn như phân biệt ảnh mèo và chó.")
        st.write("**SVM**: Tìm siêu phẳng tối ưu để phân tách các lớp dữ liệu. Phù hợp với dữ liệu tuyến tính và phi tuyến tính (với kernel như 'rbf').")
        st.write("**Neural Network**: Mô phỏng mạng nơ-ron sinh học, học các đặc trưng phức tạp qua các tầng ẩn.")
        st.write("##### Cấu trúc Neural Network")
        st.write("- **Input Layer**: Dữ liệu ảnh (64x64 pixel).")
        st.write("- **Hidden Layers**: Xử lý đặc trưng.")
        st.write("- **Output Layer**: 2 nơ-ron (Mèo hoặc Chó).")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Neural_network_example.svg/1200px-Neural_network_example.svg.png", caption="Cấu trúc mạng Neural Network", width=500)

    with tab2:
        data_dir = st.text_input("Đường dẫn thư mục dữ liệu (có thư mục con 'cats' và 'dogs'):", "path_to_data")
        if not os.path.exists(data_dir):
            st.error("Đường dẫn không tồn tại! Vui lòng nhập đúng đường dẫn.")
        else:
            n_samples = st.number_input("Số lượng mẫu", min_value=10, max_value=10000, value=100, step=10)
            X, y = load_data(data_dir, n_samples=n_samples)
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

                data_ratios = pd.DataFrame({
                    "Tập dữ liệu": ["Train", "Validation", "Test"],
                    "Tỷ lệ (%)": [train_size, val_size, test_size],
                    "Số lượng mẫu": [len(X_train), len(X_val), len(X_test)]
                })
                st.table(data_ratios)

                st.write("**🚀 Huấn luyện mô hình**")
                model_type = st.selectbox("Chọn loại mô hình", ["SVM", "Neural Network"])
                st.session_state.custom_model_name = st.text_input("Tên mô hình để lưu vào MLflow:", st.session_state.custom_model_name)

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

                st.session_state.cv_folds = st.slider("Số fold Cross-Validation", 2, 5, 3)

                if st.button("🚀 Huấn luyện"):
                    if not st.session_state.custom_model_name:
                        st.error("Vui lòng nhập tên mô hình!")
                    else:
                        with st.spinner("🔄 Đang huấn luyện..."):
                            X_train, X_val, X_test, y_train, y_val, y_test = st.session_state.data_split
                            if model_type == "SVM":
                                result = train_svm(st.session_state.custom_model_name, st.session_state.params_svm, X_train, X_val, X_test, y_train, y_val, y_test, st.session_state.cv_folds)
                                st.session_state.model_svm = result[0]
                            else:
                                result = train_nn(st.session_state.custom_model_name, st.session_state.params_nn, X_train, X_val, X_test, y_train, y_val, y_test, st.session_state.cv_folds)
                                st.session_state.model_nn = result[0]
                            st.session_state.trained_models[st.session_state.custom_model_name] = result[0]
                            st.success("✅ Huấn luyện xong!")
                            st.write(f"🎯 **Độ chính xác Train: {result[1]:.4f}**")
                            st.write(f"🎯 **Độ chính xác Validation: {result[2]:.4f}**")
                            st.write(f"🎯 **Độ chính xác Test: {result[3]:.4f}**")
                            st.write(f"🎯 **Độ chính xác CV: {result[4]:.4f}**")

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
                uploaded_file = st.file_uploader("Tải ảnh mèo/chó", type=["png", "jpg", "jpeg"])
                if uploaded_file:
                    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                    processed_image = preprocess_uploaded_image(image)
                    st.image(image, caption="Ảnh tải lên", use_column_width=True)
                    if st.button("🔮 Dự đoán"):
                        prediction = selected_model.predict(processed_image)[0]
                        probabilities = selected_model.predict_proba(processed_image)[0]
                        st.write(f"🎯 **Dự đoán: {'Mèo' if prediction == 0 else 'Chó'}**")
                        st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")
                        if show_visualization and isinstance(selected_model, MLPClassifier):
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
                        probabilities = selected_model.predict_proba(processed_canvas)[0]
                        st.write(f"🎯 **Dự đoán: {'Mèo' if prediction == 0 else 'Chó'}**")
                        st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")
                        if show_visualization and isinstance(selected_model, MLPClassifier):
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
            available_columns = [
                col for col in [
                    "model_custom_name", "params.model_name", "start_time",
                    "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy",
                    "metrics.cv_mean_accuracy"
                ] if col in runs.columns
            ]
            st.dataframe(runs[available_columns])
        else:
            st.write("⚠️ Không có phiên làm việc nào được ghi lại.")

if __name__ == "__main__":
    create_streamlit_app()
