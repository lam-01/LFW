import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 📌 Tải dữ liệu từ file CSV (tải lên hoặc mặc định)
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("flower_measurements.csv")
        except FileNotFoundError:
            st.error("Không tìm thấy file 'flower_measurements.csv'. Vui lòng tải lên file dữ liệu!")
            return None, None
    X = df.drop('Label', axis=1)
    y = df['Label'].astype(int)
    return X, y

# 📌 Chia dữ liệu thành train, validation, và test
@st.cache_data
def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (train_size + val_size), random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# 📌 Tiền xử lý dữ liệu
def preprocess_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# 📌 Hiển thị một số mẫu dữ liệu
def show_sample_data(X, y):
    st.write("**5 mẫu dữ liệu đầu tiên:**")
    sample_df = pd.concat([X, pd.Series(y, name='Label')], axis=1).head(5)
    st.dataframe(sample_df)
    
    st.write("**🌸 Minh họa vài mẫu dữ liệu**")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.scatterplot(data=pd.concat([X, pd.Series(y, name='Label')], axis=1), 
                    x='Leaf_Length', y='Petal_Size', hue='Label', palette='deep', ax=axes[0])
    axes[0].set_title("Leaf Length vs Petal Size")
    
    sns.scatterplot(data=pd.concat([X, pd.Series(y, name='Label')], axis=1), 
                    x='Stem_Length', y='Leaf_Width', hue='Label', palette='deep', ax=axes[1])
    axes[1].set_title("Stem Length vs Leaf Width")
    
    plt.tight_layout()
    st.pyplot(fig)

# 📌 Huấn luyện mô hình với K-Fold Cross Validation
def train_model(custom_model_name, model_name, params, X_train, X_val, X_test, y_train, y_val, y_test, k_folds=5):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Đang khởi tạo mô hình... (0%)")

    if model_name == "Logistic Regression":
        model = LogisticRegression(C=params["C"], max_iter=params["max_iter"], random_state=42)
    elif model_name == "SVM":
        model = SVC(kernel=params["kernel"], C=params["C"], probability=True, random_state=42)
    else:
        raise ValueError("Invalid model selected!")

    try:
        with mlflow.start_run(run_name=custom_model_name):
            # Bước 1: Khởi tạo mô hình
            progress_bar.progress(0.1)
            status_text.text("Đang thực hiện K-Fold Cross Validation... (10%)")
            start_time = time.time()

            # K-Fold Cross Validation
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            cv_scores = []
            X_train_val = np.concatenate((X_train, X_val), axis=0)
            y_train_val = np.concatenate((y_train, y_val), axis=0)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
                X_fold_train, X_fold_val = X_train_val[train_idx], X_train_val[val_idx]
                y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                y_fold_pred = model.predict(X_fold_val)
                fold_accuracy = accuracy_score(y_fold_val, y_fold_pred)
                cv_scores.append(fold_accuracy)
                progress_bar.progress(0.1 + (0.4 * (fold + 1) / k_folds))
                status_text.text(f"Cross Validation - Fold {fold + 1}/{k_folds} hoàn tất ({int(10 + 40 * (fold + 1) / k_folds)}%)")

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Bước 2: Huấn luyện mô hình trên toàn bộ tập train + validation
            progress_bar.progress(0.5)
            status_text.text("Đang huấn luyện mô hình trên toàn bộ dữ liệu... (50%)")
            model.fit(X_train_val, y_train_val)
            train_end_time = time.time()

            # Bước 3: Dự đoán trên các tập dữ liệu
            y_train_pred = model.predict(X_train)
            progress_bar.progress(0.6)
            status_text.text("Đang dự đoán trên tập train... (60%)")

            y_val_pred = model.predict(X_val)
            progress_bar.progress(0.7)
            status_text.text("Đang dự đoán trên tập validation... (70%)")

            y_test_pred = model.predict(X_test)
            progress_bar.progress(0.8)
            status_text.text("Đã dự đoán xong... (80%)")

            # Tính toán độ chính xác
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Bước 4: Ghi log vào MLflow
            status_text.text("Đang ghi log vào MLflow... (90%)")
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params)
            mlflow.log_param("k_folds", k_folds)
            mlflow.log_metric("cv_mean_accuracy", cv_mean)
            mlflow.log_metric("cv_std", cv_std)
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            
            input_example = X_train[:1]
            mlflow.sklearn.log_model(model, model_name, input_example=input_example)
            progress_bar.progress(1.0)
            status_text.text("Hoàn tất! (100%)")
    except Exception as e:
        st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
        return None, None, None, None, None, None

    return model, train_accuracy, val_accuracy, test_accuracy, cv_mean, cv_std

# 📌 Giao diện Streamlit
def create_streamlit_app():
    st.title("🌺 Phân loại hoa")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📡 Tiền xử lý dữ liệu", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])
    
    # Tab 1: Tiền xử lý dữ liệu
    with tab1:
        st.header("Tiền xử lý dữ liệu")
        
        uploaded_file = st.file_uploader("📤 Tải lên file CSV dữ liệu hoa (flower_measurements.csv)", type=["csv"])
        
        if uploaded_file is not None:
            X, y = load_data(uploaded_file)
            if X is not None and y is not None:
                st.write(f"**Kích thước dữ liệu: {X.shape}**")
                show_sample_data(X, y)
                
                st.write("**📊 Chia dữ liệu**")
                test_size = st.slider("Tỷ lệ Test (%)", min_value=5, max_value=30, value=15, step=5)
                val_size = st.slider("Tỷ lệ Validation (%)", min_value=5, max_value=30, value=15, step=5)
                
                train_size = 100 - test_size
                val_ratio = val_size / train_size
                
                if val_ratio >= 1.0:
                    st.error("Tỷ lệ Validation quá lớn so với Train! Vui lòng điều chỉnh lại.")
                else:
                    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_size=0.7, val_size=val_size/100, test_size=test_size/100)
                    if st.button("Tiền xử lý dữ liệu"):
                        X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_val, X_test)
                        st.session_state['X_train'] = X_train_scaled
                        st.session_state['X_val'] = X_val_scaled
                        st.session_state['X_test'] = X_test_scaled
                        st.session_state['y_train'] = y_train
                        st.session_state['y_val'] = y_val
                        st.session_state['y_test'] = y_test
                        st.session_state['scaler'] = scaler
                        
                        data_ratios = pd.DataFrame({
                            "Tập dữ liệu": ["Train", "Validation", "Test"],
                            "Tỷ lệ (%)": [train_size - val_size, val_size, test_size],
                            "Số lượng mẫu": [len(X_train), len(X_val), len(X_test)]
                        })
                        st.table(data_ratios)
                        st.success("Tiền xử lý dữ liệu hoàn tất!")
        else:
            st.info("Vui lòng tải lên file CSV để bắt đầu tiền xử lý dữ liệu.")

    # Tab 2: Huấn luyện (bổ sung K-Fold)
    with tab2:
        st.header("Huấn luyện mô hình")
        if 'X_train' not in st.session_state:
            st.warning("Vui lòng thực hiện tiền xử lý dữ liệu trước!")
        else:
            custom_model_name = st.text_input("Nhập tên mô hình:", "")
            if not custom_model_name:
                custom_model_name = "Default_model"

            model_name = st.selectbox("🔍 Chọn mô hình", ["Logistic Regression", "SVM"])
            params = {}

            if model_name == "Logistic Regression":
                params["C"] = st.slider("🔧 Tham số C", 0.1, 10.0, 1.0)
                params["max_iter"] = st.slider("🔄 Số lần lặp tối đa", 100, 1000, 100, step=100)
            elif model_name == "SVM":
                params["kernel"] = st.selectbox("⚙️ Kernel", ["linear", "rbf", "poly", "sigmoid"])
                params["C"] = st.slider("🔧 Tham số C", 0.1, 10.0, 1.0)

            # Thêm tùy chọn K-Fold
            k_folds = st.slider("🔢 Số lượng K-Fold cho Cross Validation", min_value=2, max_value=10, value=5, step=1)

            if st.button("🚀 Huấn luyện mô hình"):
                with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                    model, train_accuracy, val_accuracy, test_accuracy, cv_mean, cv_std = train_model(
                        custom_model_name, model_name, params, 
                        st.session_state['X_train'], st.session_state['X_val'], st.session_state['X_test'],
                        st.session_state['y_train'], st.session_state['y_val'], st.session_state['y_test'],
                        k_folds=k_folds
                    )
                
                if model is not None:
                    st.session_state['model'] = model
                    st.success(f"✅ Huấn luyện xong!")
                    st.write(f"🎯 **Độ chính xác Cross Validation (mean ± std): {cv_mean:.4f} ± {cv_std:.4f}**")
                    st.write(f"🎯 **Độ chính xác trên tập train: {train_accuracy:.4f}**")
                    st.write(f"🎯 **Độ chính xác trên tập validation: {val_accuracy:.4f}**")
                    st.write(f"🎯 **Độ chính xác trên tập test: {test_accuracy:.4f}**")
                else:
                    st.error("Huấn luyện thất bại, không có kết quả để hiển thị.")

    # Tab 3: Dự đoán
    with tab3:
        st.header("Dự đoán")
        
        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty and 'scaler' in st.session_state:
            runs["model_custom_name"] = runs["tags.mlflow.runName"]
            model_names = runs["model_custom_name"].tolist()
            
            selected_model_name = st.selectbox("🔍 Chọn mô hình để dự đoán", model_names)
            if selected_model_name:
                selected_run = runs[runs["model_custom_name"] == selected_model_name].iloc[0]
                model_uri = f"runs:/{selected_run['run_id']}/{selected_run['params.model_name']}"
                try:
                    selected_model = mlflow.sklearn.load_model(model_uri)
                    
                    st.write("Nhập thông số hoa để dự đoán:")
                    leaf_length = st.number_input("Leaf Length", min_value=0.0, value=5.0)
                    leaf_width = st.number_input("Leaf Width", min_value=0.0, value=2.0)
                    stem_length = st.number_input("Stem Length", min_value=0.0, value=30.0)
                    petal_size = st.number_input("Petal Size", min_value=0.0, value=3.0)
                    
                    if st.button("🔮 Dự đoán"):
                        input_data = np.array([[leaf_length, leaf_width, stem_length, petal_size]])
                        input_scaled = st.session_state['scaler'].transform(input_data)
                        prediction = selected_model.predict(input_scaled)[0]
                        probabilities = selected_model.predict_proba(input_scaled)[0]
                        st.write(f"🎯 **Dự đoán: Label {prediction}**")
                        st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")
                except Exception as e:
                    st.error(f"Không thể tải mô hình: {str(e)}")
        else:
            st.warning("Vui lòng huấn luyện ít nhất một mô hình và thực hiện tiền xử lý dữ liệu trước!")

    # Tab 4: MLflow
    with tab4:
        st.header("MLflow Tracking")
        st.write("Xem chi tiết các kết quả đã lưu trong MLflow.")
        
        runs = mlflow.search_runs(order_by=["start_time desc"])
        if not runs.empty:
            runs["model_custom_name"] = runs["tags.mlflow.runName"]
            
            search_model_name = st.text_input("🔍 Nhập tên mô hình để tìm kiếm:", "")
            if search_model_name:
                filtered_runs = runs[runs["model_custom_name"].str.contains(search_model_name, case=False, na=False)]
            else:
                filtered_runs = runs

            if not filtered_runs.empty:
                st.write("📜 Danh sách mô hình đã lưu:")
                available_columns = [col for col in ["model_custom_name", "params.model_name", "start_time", 
                                                     "metrics.train_accuracy", "metrics.val_accuracy", 
                                                     "metrics.test_accuracy", "metrics.cv_mean_accuracy"] 
                                     if col in runs.columns]
                display_df = filtered_runs[available_columns]
                
                for col in display_df.columns:
                    if display_df[col].dtype == 'object':
                        display_df[col] = display_df[col].astype(str)
                
                display_df = display_df.rename(columns={
                    "model_custom_name": "Custom Model Name",
                    "params.model_name": "Model Type",
                    "metrics.cv_mean_accuracy": "CV Mean Accuracy"
                })
                st.dataframe(display_df)

                selected_model_name = st.selectbox("📝 Chọn một mô hình để xem chi tiết:", 
                                                   filtered_runs["model_custom_name"].tolist())
                if selected_model_name:
                    selected_run = filtered_runs[filtered_runs["model_custom_name"] == selected_model_name].iloc[0]
                    run_details = mlflow.get_run(selected_run["run_id"])
                    custom_name = run_details.data.tags.get('mlflow.runName', 'Không có tên')
                    model_type = run_details.data.params.get('model_name', 'Không xác định')
                    st.write(f"🔍 Chi tiết mô hình: `{custom_name}`")
                    st.write(f"**📌 Loại mô hình huấn luyện:** {model_type}")

                    st.write("📌 **Tham số:**")
                    for key, value in run_details.data.params.items():
                        if key != 'model_name':
                            st.write(f"- **{key}**: {value}")

                    st.write("📊 **Metric:**")
                    for key, value in run_details.data.metrics.items():
                        st.write(f"- **{key}**: {value}")
            else:
                st.write("❌ Không tìm thấy mô hình nào.")
        else:
            st.write("⚠️ Không có phiên làm việc nào được ghi lại.")

if __name__ == "__main__":
    create_streamlit_app()
