import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Changed this line
import pandas as pd

import altair as alt
import textwrap
# Cấu hình giao diện
st.set_page_config(page_title="Phân loại lá cây", layout="centered")
st.markdown("<h1 style='text-align: center; color: green;'>🌿 Phân loại Lá Cây - Phát hiện Bệnh (MobileNetV2)</h1>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("best_model_mobile.keras")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_trained_model()

# Load danh sách lớp
@st.cache_data
def load_class_names():
    try:
        with open("class_names.txt", "r") as f:
            raw_names = [line.strip() for line in f.readlines()]
        
        # Gán nhãn tiếng Việt
        mapping = {
            "Pepper__bell___Bacterial_spot": "Ớt chuông - Đốm vi khuẩn (Bacterial spot)",
            "Pepper__bell___healthy": "Ớt chuông - Khỏe mạnh (healthy)",
            "Potato___Early_blight": "Khoai tây - Bệnh sớm (Early blight)",
            "Potato___Late_blight": "Khoai tây - Bệnh sương mai (Late blight)",
            "Potato___healthy": "Khoai tây - Khỏe mạnh (healthy)",
            "Tomato_Bacterial_spot": "Cà chua - Đốm vi khuẩn (Bacterial spot)",
            "Tomato_Early_blight": "Cà chua - Bệnh sớm (Early blight)",
            "Tomato_Late_blight": "Cà chua - Bệnh sương mai (Late blight)",
            "Tomato_Leaf_Mold": "Cà chua - Mốc lá (Leaf Mold)",
            "Tomato_Septoria_leaf_spot": "Cà chua - Đốm lá Septoria (Septoria leaf spot)",
            "Tomato_Spider_mites_Two_spotted_spider_mite": "Cà chua - Nhện đỏ hai đốm (Two-spotted spider mite)",
            "Tomato__Target_Spot": "Cà chua - Đốm mục tiêu (Target Spot)",
            "Tomato__Tomato_YellowLeaf__Curl_Virus": "Cà chua - Virus xoăn vàng lá (Tomato Yellow Leaf Curl Virus)",
            "Tomato__Tomato_mosaic_virus": "Cà chua - Virus khảm (Tomato mosaic virus)",
            "Tomato_healthy": "Cà chua - Khỏe mạnh (healthy)"
        }

        return [mapping.get(name, name) for name in raw_names]
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        return []

class_names = load_class_names()

# Hàm tiền xử lý ảnh
def preprocess_image(img, img_size=(256, 256)):  # MobileNetV2 standard size
    try:
        img = img.convert("RGB")
        img = img.resize(img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Thêm batch dimension
        return preprocess_input(img_array)  # MobileNetV2 preprocessing
    except Exception as e:
        st.error(f"❌ Lỗi xử lý ảnh: {e}")
        return None

# Hàm dự đoán
def predict_with_probabilities(model, img_array):
    try:
        predictions = model.predict(img_array, verbose=0)[0] # Lấy dự đoán cho batch đầu tiên
        predicted_index = np.argmax(predictions)  # Class có xác suất cao nhất
        predicted_class = class_names[predicted_index] # Lấy tên lớp tương ứng
        confidence = float(predictions[predicted_index]) * 100  # Chuyển đổi sang phần trăm
        return predicted_class, confidence, predictions
    except Exception as e:
        st.error(f"❌ Lỗi dự đoán: {e}")
        return None, 0, []

# Hiển thị thông tin model
if model and class_names:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "MobileNetV2")
    with col2:
        st.metric("Classes", len(class_names))
    with col3:
        st.metric("Input Size", "256x256")

# Upload ảnh
uploaded_file = st.file_uploader("📤 Tải lên ảnh lá cây", type=["jpg", "jpeg", "png"])

if uploaded_file and model and class_names:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Hiển thị ảnh gốc và ảnh đã resize
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="📷 Ảnh gốc", use_column_width=True)
    with col2:
        resized_img = img.resize((256, 256))
        st.image(resized_img, caption="🔄 Ảnh đã resize (256x256)", use_column_width=True)

    if st.button("📊 Phân tích", type="primary"):
        with st.spinner("⏳ Đang phân tích..."):
            processed_img = preprocess_image(img)
            if processed_img is not None:
                result = predict_with_probabilities(model, processed_img)
                if result[0] is not None:
                    predicted_class, confidence, probs = result

                    # Kết quả chính
                    if confidence > 80:
                        confidence_color = "green"
                        confidence_icon = "🟢"
                    elif confidence > 60:
                        confidence_color = "orange"
                        confidence_icon = "🟡"
                    else:
                        confidence_color = "red"
                        confidence_icon = "🔴"

                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: black;'>
                        <h3>🔍 Kết quả dự đoán:</h3>
                        <h2 style='color: blue;'>{predicted_class}</h2>
                        <p>{confidence_icon} Độ tin cậy: <span style='color: {confidence_color}; font-weight: bold;'>{confidence:.2f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(confidence / 100)

                    # Xác suất top 5
                    st.markdown("<h4>🔎 Top 5 xác suất cao nhất:</h4>", unsafe_allow_html=True)
                    top_indices = np.argsort(probs)[::-1][:5]
                    top_probs_df = pd.DataFrame({
                        "Thứ hạng": range(1, 6),
                        "Lớp": [class_names[i] for i in top_indices],
                        "Xác suất (%)": [round(probs[i] * 100, 2) for i in top_indices]
                    })
                    st.dataframe(top_probs_df)
                    
                    
                    chart_df = pd.DataFrame({
                        "Class": [class_names[i] for i in top_indices],
                        "Probability": [probs[i] * 100 for i in top_indices]
                    })
                    st.altair_chart(
                        alt.Chart(chart_df).mark_bar(color="#90cdf4").encode(
                            x=alt.X(
                                "Class:N",
                                sort="-y",
                                axis=alt.Axis(        
                                    labelAngle=0,
                                    labelLimit=120      
                                )
                            ),
                            y="Probability:Q"
                        ),
                        use_container_width=True
                    )

elif not model:
    st.error("❌ Không thể tải model. Vui lòng kiểm tra file best_model_mobile.keras")
elif not class_names:
    st.error("❌ Không thể tải danh sách lớp. Vui lòng kiểm tra file class_names.txt")
else:
    st.info("🖼️ Vui lòng tải lên một ảnh để bắt đầu.")

# Footer
st.markdown("<hr><div style='text-align: center;'>🌱 Ứng dụng kiểm tra sức khỏe cây trồng với MobileNetV2</div>", unsafe_allow_html=True)