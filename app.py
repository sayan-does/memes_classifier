import streamlit as st
from PIL import Image
import torch
import numpy as np
from utility import preprocess_image, load_model, generate_gradcam

# Streamlit app
st.set_page_config(page_title="Check a Meme",
                   page_icon=":smile:", layout="wide")

# Apply custom CSS for a background image and other styles


# def add_custom_css():
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background-image: url("G:\Projectsdocs\memes\csspatternbymagicpattern.jpeg");
#             background-size: cover;
#             color: white;
#         }

#         .prediction {
#             font-size: 24px;
#             font-weight: bold;
#         }

#         .prediction.not-meme {
#             color: #8B0000;
#         }

#         .image-container {
#             text-align: center;
#             margin-bottom: 20px;
#         }

#         .image-container img {
#             max-width: 80%;  /* Adjust this value to reduce the image size */
#             height: auto;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

def add_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #272728;
            opacity: 0.8;
            background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #272728 36px ), repeating-linear-gradient( #1c82a155, #1c82a1 );
            color: white;
        }

        .prediction {
            font-size: 24px;
            font-weight: bold;
        }

        .prediction.not-meme {
            color: #8B0000;
        }

        .image-container {
            text-align: center;
            margin-bottom: 20px;
        }

        .image-container img {
            max-width: 80%;  /* Adjust this value to reduce the image size */
            height: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


add_custom_css()



add_custom_css()

st.title("Check a Meme")
st.subheader(
    "Memes are humorous images, videos, text, etc. A meme is a virally transmitted image embellished with text, usually sharing pointed commentary on cultural symbols, social ideas, or current events. If you have a slow meme brain, lend my model to know a image is a meme or not!")

# Column layout for file uploader and results
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Drag and Drop image", type=[
                                     "jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    image_tensor = preprocess_image(uploaded_file)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model('best_model.pth', device)

    # Predict the class
    model.eval()
    output = model(image_tensor.to(device))
    _, predicted = torch.max(output, 1)
    class_idx = predicted.item()
    label = 'Meme' if class_idx == 0 else 'Not Meme'

    # Generate Grad-CAM visualization
    target_layer = model.features.denseblock4
    gradcam_img = generate_gradcam(model, image_tensor, device, target_layer)

    # Convert Grad-CAM and original image to displayable format
    gradcam_img_pil = Image.fromarray((gradcam_img * 255).astype(np.uint8))

    # Display the prediction
    with col1:
        if label == 'Meme':
            st.markdown(
                f'<p class="prediction">Prediction: {label} ✅</p>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<p class="prediction not-meme">Prediction: {label}</p>', unsafe_allow_html=True)

        # Display Grad-CAM color explanation
        st.markdown("""
        **Grad-CAM Color Explanation:**

        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 20px; background-color: red; margin-right: 5px;"></div> Red: High importance regions (strong influence on the prediction)
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 20px; background-color: orange; margin-right: 5px;"></div> Yellow/Orange: Moderate importance regions
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 20px; background-color: blue; margin-right: 5px;"></div> Blue/Green: Low importance regions (weak influence on the prediction)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption='Original Image', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(gradcam_img_pil,
                 caption='Grad-CAM heatmap of image features responsible for the prediction', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
