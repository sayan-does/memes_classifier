import streamlit as st
from PIL import Image
import torch
import numpy as np
from utility import preprocess_image, load_model, generate_gradcam
import base64

# Streamlit app
st.set_page_config(page_title="Check a Meme",
                   page_icon=":smile:", layout="wide")

# Function to set the background image


def set_background(png_file):
    with open(png_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Replace 'background.png' with the path to your background image file
set_background('css-pattern-by-magicpattern (3).png')

# Function to set custom header font


def set_header_font_and_border():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');
        .stApp h1 {
            font-family: 'Playfair Display', serif;
            font-size: 3em;
            color: #00000e;
            font-weight: bold;
        }
        .stApp h3 {

            color: #00000e;
            font-size: 1.5em;
            font-weight: bold;
            
        }
        .stApp p {

            color: #00000e;
            font-size: 1em;
            font-weight: bold;
            
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Set custom header font
set_header_font_and_border()

st.title("Check a Meme")
st.write("### What is a Meme?")
st.write("""
Memes are humorous images, videos, text, etc. A meme is a virally transmitted image embellished with text,
usually sharing pointed commentary on cultural symbols, social ideas, or current events.
If you have a slow meme brain, lend my model to know a image is a meme or not!
""")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=[
                                     "jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    with col2:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        model = load_model('best_model.pt', device)

        # Preprocess image
        image_tensor = preprocess_image(uploaded_file)

        # Predict the class
        model.eval()
        output = model(image_tensor.to(device))
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        label = 'Meme' if class_idx == 0 else 'Not Meme'

        # Display the prediction
        st.markdown(f"**Prediction:** {label}")

        # Generate Grad-CAM visualization
        target_layer = model.features.denseblock4
        gradcam_img = generate_gradcam(
            model, image_tensor, device, target_layer)

        # Convert Grad-CAM and original image to displayable format
        original_img = Image.open(uploaded_file).convert('RGB')
        gradcam_img_pil = Image.fromarray((gradcam_img * 255).astype(np.uint8))

        # Display the images side by side
        col3, col4 = st.columns(2)
        with col3:
            st.image(original_img, caption='Original Image',
                     use_column_width=True)
        with col4:
            st.image(gradcam_img_pil, caption='Grad-CAM heatmap of image features responsible for the prediction',
                     use_column_width=True)
