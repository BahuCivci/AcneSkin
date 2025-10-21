from io import BytesIO
import streamlit as st
from PIL import Image

def uploader():
    st.title("Upload Your Selfie for Analysis")
    
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert the image to bytes for further processing
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
    return None