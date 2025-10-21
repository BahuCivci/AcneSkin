import streamlit as st
from src.components.uploader import image_uploader
from src.ai.predict import analyze_skin
from src.utils.preprocessing import preprocess_image

def upload_page():
    st.title("Upload Your Selfie for Skin Analysis")
    
    # Image uploader component
    uploaded_image = image_uploader()
    
    if uploaded_image is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_image)
        
        # Analyze the skin using the AI model
        analysis_results = analyze_skin(preprocessed_image)
        
        # Display the results
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("### Analysis Results:")
        st.json(analysis_results)

if __name__ == "__main__":
    upload_page()