from streamlit import st

def main():
    st.title("Welcome to Acne AI – Skin Digital Twin")
    st.write("This application provides AI-based skin analysis and personalized skincare recommendations.")
    
    st.header("Features")
    st.write("- **AI-based Skin Analysis**: Upload your photo to get an analysis of your skin.")
    st.write("- **Skin Passport Report**: Receive a detailed report summarizing your skin condition and tips for improvement.")
    
    st.header("Get Started")
    st.write("To begin, navigate to the upload page to submit your selfie for analysis.")
    
    if st.button("Go to Upload"):
        st.session_state.page = "upload"
        st.experimental_rerun()

if __name__ == "__main__":
    main()