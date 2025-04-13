import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
import joblib
import os

# Page settings
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 Lung Cancer Detection Web App")
st.markdown("Upload a lung CT image and choose a model to predict the cancer type.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload a CT scan image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# Model selection
model_choice = st.radio("🧠 Select a model to use:", ["CNN (Deep Learning)", "SVM (Machine Learning)"])

# Class names used by both models
class_names_CNN = ['Adenocarcinoma', 'Normal', 'Squamous Cell Carcinoma']

# CNN Prediction Function
def predict_with_cnn(img_array):
    try:
        model = tf.keras.models.load_model("lung_cancer_model.h5")
        cnn_input = img_array / 255.0
        cnn_input = np.expand_dims(cnn_input, axis=0)
        prediction = model.predict(cnn_input)
        predicted_label = class_names_CNN[np.argmax(prediction)]
        return f"✅ CNN Prediction: **{predicted_label}**"
    except Exception as e:
        return f"❌ Error loading CNN model: {e}"

# SVM Prediction Function
class_names_SVM = ['Squamous Cell Carcinoma', 'Normal', 'Adenocarcinoma']
def predict_with_svm(uploaded_image):
    try:
        if not os.path.exists("svm_model.pkl"):
            return "❌ Error: 'svm_model.pkl' not found."

        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return "❌ Error: Could not read image."

        img_resized = cv2.resize(img, (64, 64))

        # Use same HOG config as training
        features, _ = hog(img_resized, 
                          pixels_per_cell=(8, 8), 
                          cells_per_block=(2, 2), 
                          visualize=True)

        features = features.reshape(1, -1)  # Shape (1, 1764)

        model = joblib.load("svm_model.pkl")
        prediction = model.predict(features)
        predicted_label = class_names_SVM[int(prediction[0])]
        return f"✅ SVM Prediction: **{predicted_label}**"
    except Exception as e:
        return f"❌ Error using SVM model: {e}"

# Main Prediction Logic
if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
    st.markdown("---")
    
    if st.button("🔍 Predict"):
        try:
            img = Image.open(uploaded_file).convert('RGB')
            img_resized = img.resize((128, 128))
            img_array = np.array(img_resized)

            if model_choice == "CNN (Deep Learning)":
                result = predict_with_cnn(img_array)

            elif model_choice == "SVM (Machine Learning)":
                # Pass uploaded_file again because it's read inside the function
                uploaded_file.seek(0)  # Reset stream pointer
                result = predict_with_svm(uploaded_file)

            st.success(result if result.startswith("✅") else "")
            if result.startswith("❌"):
                st.error(result)
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

        st.markdown("---")
        st.info("📌 *Note: For best results, use clear and centered CT scan images.*")
else:
    st.warning("⚠️ Please upload an image to begin prediction.")
