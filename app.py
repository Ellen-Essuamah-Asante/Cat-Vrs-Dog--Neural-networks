# Page Config 
st.set_page_config(
    page_title="Cat vs Dog Classifier 🐾",
    page_icon="🐶",
    layout="centered",
)

# Custom Styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f7f7;
            padding: 2rem;
            border-radius: 15px;
        }
        .stButton>button {
            background-color: #FF6F61;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #ff3b2f;
            transform: scale(1.03);
        }
        .stImage img {
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Title and Description 
st.title("🐾 Cat vs Dog Classifier")
st.markdown("""
Upload an image and let our AI-powered model guess whether it's a **cat** 🐱 or a **dog** 🐶.

Built with **TensorFlow** + **Streamlit**, this app uses a CNN model trained on real cat & dog images.
""")

# Load Model 
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cat_dog_model.h5")
    return model

model = load_model()

#  File Upload 
st.subheader("📸 Upload Your Image")
uploaded_file = st.file_uploader("Drag & drop or select an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="✨ Your Uploaded Image", use_container_width=True)
    st.markdown("---")
    
    # Predict Button
    if st.button("🔍 Classify Image"):
        with st.spinner("Analyzing image... please wait ⏳"):
            img_resized = img.resize((150, 150))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)[0][0]

            st.markdown("---")
            if prediction > 0.5:
                st.success("### 🐶 It’s a **Dog!**")
                st.balloons()
            else:
                st.success("### 🐱 It’s a **Cat!**")
                st.snow()

            st.markdown(f"**Prediction Confidence:** `{(prediction if prediction > 0.5 else 1 - prediction)*100:.2f}%`")

else:
    st.info("👆 Upload an image above to get started!")

# --- Footer ---
st.markdown("""
<hr>
<div style='text-align:center; color:gray; font-size:0.9em;'>
Made with ❤️ using Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)
