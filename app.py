# app.py

import streamlit as st
from chatbot import get_response, predict_image
from tensorflow.keras.utils import load_img

st.set_page_config(page_title="Multimodal COVID-19 Chatbot", layout="wide")
st.title("🦠 Multimodal COVID-19 Detection & Chatbot")

# --- Chatbot Section ---
st.sidebar.title("🤖 Ask the Chatbot")
user_input = st.sidebar.text_input("Ask something about COVID-19")

if user_input:
    response = get_response(user_input)
    st.sidebar.markdown(f"**Bot:** {response}")

# --- Image Upload & Prediction Section ---
st.subheader("📷 Upload Chest X-ray for COVID-19 Detection")

uploaded_file = st.file_uploader("Upload a Chest X-ray (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Chest X-ray", use_column_width=False, width=300)

    # Load and preprocess image
    image = load_img(uploaded_file, target_size=(256, 256))

    label, confidence = predict_image(image)

    st.markdown(f"### 🧪 Prediction: **{label}**")
    st.markdown(f"*Confidence Score: `{confidence:.4f}`*")
