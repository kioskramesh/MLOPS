import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("🏡 California House Price Predictor")
st.write("This AI model predicts the median house value based on the California Housing dataset.")

# 1. Load the frozen model artifact
@st.cache_resource
def load_model():
    return joblib.load("model/model.joblib")

model = load_model()

st.write("### Enter Property Details:")

# 2. Add input sliders matching features from the dataset
med_inc = st.slider("Median Income (in tens of thousands, e.g., 3.5 = $35,000)", 0.5, 15.0, 3.5)
house_age = st.slider("Median House Age", 1, 52, 28)
ave_rooms = st.slider("Average Rooms per Household", 1.0, 10.0, 5.0)
ave_bedrms = st.slider("Average Bedrooms per Household", 0.5, 5.0, 1.1)
population = st.slider("Block Population", 3, 35000, 1400)
ave_occup = st.slider("Average House Occupants", 1.0, 10.0, 3.0)
latitude = st.slider("Latitude", 32.5, 42.5, 35.5)
longitude = st.slider("Longitude", -124.3, -114.3, -119.5)

# 3. Create input DataFrame matching feature names exactly
input_data = pd.DataFrame([{
    'MedInc': med_inc,
    'HouseAge': house_age,
    'AveRooms': ave_rooms,
    'AveBedrms': ave_bedrms,
    'Population': population,
    'AveOccup': ave_occup,
    'Latitude': latitude,
    'Longitude': longitude
}])

# 4. Predict button
if st.button("Predict House Value"):
    prediction = model.predict(input_data)
    # The dataset target is in hundreds of thousands ($100,000s)
    final_price = float(prediction[0]) * 100000
    st.success(f"### 💰 Estimated Value: ${final_price:,.2f}")
