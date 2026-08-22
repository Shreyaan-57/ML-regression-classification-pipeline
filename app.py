import streamlit as st
import joblib
import numpy as np

# Load serialized model and scaler
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('cls_scaler.pkl')

st.title("Breast Cancer Diagnostic Tool")
st.write("Enter feature values to predict diagnosis.")

# Example input fields (adjust features based on dataset)
radius = st.number_input("Mean Radius", value=14.0)
texture = st.number_input("Mean Texture", value=19.0)

if st.button("Run Prediction"):
    # Create dummy array matching model's expected shape (30 features)
    sample_data = np.zeros((1, 30))
    sample_data[0, 0] = radius
    sample_data[0, 1] = texture
    
    # Scale and predict
    scaled_data = scaler.transform(sample_data)
    prediction = model.predict(scaled_data)[0]
    
    if prediction == 0:
        st.error("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")