import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="🐾 Cat & Dog Classifier", page_icon="🐶", layout="wide")

st.title("🐱🐶 Cat & Dog Image Classifier")
st.write("Upload an image to find out whether it's a cat or a dog!")

@st.cache_resource
def load_model():
    # 👇 Update this path to point to your Google Drive model
    model = tf.keras.models.load_model("/content/drive/MyDrive/cat_dog_model.h5")
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))  # adjust if your model uses a different size
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    result = "🐱 Cat" if prediction[0][0] < 0.5 else "🐶 Dog"

    st.subheader(f"Prediction: {result}")
