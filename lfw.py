import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    
    st.write("**🌸 Biểu đồ phân bố dữ liệu**")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=pd.concat([X, pd.Series(y, name='Label')], axis=1), 
                   x='Leaf_Length', y='Petal_Size', hue='Label', palette='deep')
    ax.set_title("Phân bố Leaf Length và Petal Size theo Label")
    st.pyplot(fig)

# 📌 Giao diện Streamlit
def create_streamlit_app():
    st.title("🌺 Phân loại hoa")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📡 Tiền xử lý dữ liệu", "📋 Huấn luyện", "🔮 Dự đoán", "⚡ MLflow"])
    
    # Tab 1: Tiền xử lý dữ liệu
    with tab1:
        st.header("Tiền xử lý dữ liệu")
        
        # Upload file CSV
        uploaded_file = st.file_uploader("📤 Tải lên file CSV dữ liệu hoa (flower_measurements.csv)", type=["csv"])
        
        if uploaded_file is not None:
            X, y = load_data(uploaded_file)
            if X is not None and y is not None:
                # Hiển thị kích thước dữ liệu
                st.write(f"**Kích thước dữ liệu: {X.shape}**")
                
                # Hiển thị 5 mẫu dữ liệu đầu tiên và biểu đồ
                show_sample_data(X, y)
                
                # Chia dữ liệu
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

    # Các tab khác giữ nguyên (Tab 2, Tab 3, Tab 4)
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

            if st.button("🚀 Huấn luyện mô hình"):
                with st.spinner("🔄 Đang khởi tạo huấn luyện..."):
                    model, train_accuracy, val_accuracy, test_accuracy = train_model(
                        custom_model_name, model_name, params, 
                        st.session_state['X_train'], st.session_state['X_val'], st.session_state['X_test'],
                        st.session_state['y_train'], st.session_state['y_val'], st.session_state['y_test']
                    )
                
                if model is not None:
                    st.session_state['model'] = model
                    st.success(f"✅ Huấn luyện xong!")
                    st.write(f"🎯 **Độ chính xác trên tập train: {train_accuracy:.4f}**")
                    st.write(f"🎯 **Độ chính xác trên tập validation: {val_accuracy:.4f}**")
                    st.write(f"🎯 **Độ chính xác trên tập test: {test_accuracy:.4f}**")
                else:
                    st.error("Huấn luyện thất bại, không có kết quả để hiển thị.")

    with tab3:
        st.header("Dự đoán")
        if 'model' not in st.session_state or 'scaler' not in st.session_state:
            st.warning("Vui lòng huấn luyện mô hình trước!")
        else:
            st.write("Nhập thông số hoa để dự đoán:")
            leaf_length = st.number_input("Leaf Length", min_value=0.0, value=5.0)
            leaf_width = st.number_input("Leaf Width", min_value=0.0, value=2.0)
            stem_length = st.number_input("Stem Length", min_value=0.0, value=30.0)
            petal_size = st.number_input("Petal Size", min_value=0.0, value=3.0)
            
            if st.button("🔮 Dự đoán"):
                input_data = np.array([[leaf_length, leaf_width, stem_length, petal_size]])
                input_scaled = st.session_state['scaler'].transform(input_data)
                prediction = st.session_state['model'].predict(input_scaled)[0]
                probabilities = st.session_state['model'].predict_proba(input_scaled)[0]
                st.write(f"🎯 **Dự đoán: Label {prediction}**")
                st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

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
                                                     "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy"] 
                                     if col in runs.columns]
                display_df = filtered_runs[available_columns]
                
                for col in display_df.columns:
                    if display_df[col].dtype == 'object':
                        display_df[col] = display_df[col].astype(str)
                
                display_df = display_df.rename(columns={
                    "model_custom_name": "Custom Model Name",
                    "params.model_name": "Model Type"
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
