import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('models/waste_sorting_model_final.keras')

# Waste categories (ensure these match the actual class names from your dataset)
categories = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Preprocess the uploaded image to fit the model input
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image to 224x224
    img_array = image.img_to_array(img)  # Convert the image to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims to add batch size
    img_array /= 255.0  # Normalize the pixel values
    return img_array

# Streamlit App Title and UI
st.set_page_config(page_title="AI-Powered Waste Sorting", page_icon="♻️")

st.title("♻️ AI-Powered Smart Waste Sorting System")
st.write("Upload an image of a waste item, and the AI will classify its type.")

# File uploader widget to allow user to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess and make prediction
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]  # Get the class label

    # Display the result
    st.success(f"The waste item is predicted to be: **{predicted_class}**")

# Footer with custom style
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    footer:after {
        content:'Created by Mohammed Madany | AI-Powered Smart Waste Sorting System';
        visibility: visible;
        display: block;
        position: relative;
        padding: 10px;
        color: gray;
        top: 3px;
    }
    </style>
    """, unsafe_allow_html=True)