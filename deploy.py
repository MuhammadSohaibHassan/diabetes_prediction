import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# Load the saved Random Forest model
model = joblib.load('random_forest_model.pkl')

# Define the feature names for input
FEATURES = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# Class labels
CLASS_LABELS = {1: 'Diabetic (Y)', 0: 'Non-Diabetic (N)', 2: 'Pre-Diabetic (P)'}

# Streamlit app
st.title("Diabetes Prediction App")
st.write("This app predicts the likelihood of being Diabetic, Non-Diabetic, or Pre-Diabetic based on health metrics.")

# Sidebar for inputs
st.sidebar.header("Input Health Metrics")
def user_input_features():
    input_data = {}
    input_data['AGE'] = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=50, step=1)
    input_data['Gender'] = st.sidebar.radio("Gender", ["Male", "Female"], index=0)
    input_data['Urea'] = st.sidebar.number_input("Urea (mg/dl)", value=4.7, step=0.1)
    input_data['Cr'] = st.sidebar.number_input("Creatinine (mg/dl)", value=46.0, step=0.1)
    input_data['HbA1c'] = st.sidebar.number_input("HbA1c (%)", value=4.9, step=0.1)
    input_data['Chol'] = st.sidebar.number_input("Cholesterol (mg/dl)", value=4.2, step=0.1)
    input_data['TG'] = st.sidebar.number_input("Triglycerides (mg/dl)", value=0.9, step=0.1)
    input_data['HDL'] = st.sidebar.number_input("HDL (mg/dl)", value=2.4, step=0.1)
    input_data['LDL'] = st.sidebar.number_input("LDL (mg/dl)", value=1.4, step=0.1)
    input_data['VLDL'] = st.sidebar.number_input("VLDL (mg/dl)", value=0.5, step=0.1)
    input_data['BMI'] = st.sidebar.number_input("BMI (kg/m²)", value=24.0, step=0.1)
    return pd.DataFrame([input_data])

data = user_input_features()

# Display the input data
st.subheader("Your Input Data")
styled_data = data.copy()
styled_data['Gender'] = styled_data['Gender'].apply(lambda x: 1 if x == "Male" else 0)  # Encode gender for the model
st.write(data)

prediction = model.predict(styled_data[FEATURES])[0]
prediction_proba = model.predict_proba(styled_data[FEATURES])[0] * 100  # Convert to percentages

# Display prediction results
st.subheader("Prediction Result")
st.write(f"Predicted Status: **{CLASS_LABELS[prediction]}**")

st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame([prediction_proba], columns=[CLASS_LABELS[i] for i in range(len(CLASS_LABELS))])
st.write(proba_df.style.format("{:.2f}%"))

# Plot prediction probabilities as a pie chart
fig, ax = plt.subplots()
ax.pie(prediction_proba, labels=[CLASS_LABELS[i] for i in range(len(CLASS_LABELS))], autopct="%.1f%%", colors=["#ff9999", "#66b3ff", "#99ff99"])
ax.set_title("Prediction Probabilities")

# Display the pie chart
buf = BytesIO()
plt.savefig(buf, format="png")
st.image(buf)
buf.close()

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit.")
