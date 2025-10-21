from streamlit import st
import pandas as pd
from src.services.storage import get_user_data
from src.utils.metrics import calculate_skin_metrics

def generate_skin_passport(user_id):
    # Fetch user data and analysis results
    user_data = get_user_data(user_id)
    skin_metrics = calculate_skin_metrics(user_data['analysis_results'])

    # Create a DataFrame for the report
    passport_data = {
        "Metric": list(skin_metrics.keys()),
        "Value": list(skin_metrics.values())
    }
    passport_df = pd.DataFrame(passport_data)

    return passport_df

def display_skin_passport():
    st.title("Your Skin Passport")
    
    user_id = st.session_state.get('user_id')
    
    if user_id:
        passport_df = generate_skin_passport(user_id)
        st.write("### Analysis Summary")
        st.dataframe(passport_df)

        st.write("### Personalized Tips")
        st.write("1. Stay hydrated.")
        st.write("2. Use sunscreen daily.")
        st.write("3. Follow a consistent skincare routine.")
    else:
        st.warning("Please upload a photo to generate your Skin Passport.")

if __name__ == "__main__":
    display_skin_passport()