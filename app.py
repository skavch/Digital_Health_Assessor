import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Load Dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('health_vitals_dataset.csv')  # Load your dataset here

# Preprocessing
X = data.drop('health_state', axis=1)
y = data['health_state']

# Encode target variable
y = y.astype('category').cat.codes  # convert target to categorical codes

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training using Support Vector Machine (SVM)
model = SVC(kernel='linear', probability=True, random_state=42)  # Enable probability estimates
model.fit(X_train, y_train)

# Ideal ranges for health parameters
ideal_ranges = {
    "heart_rate": (60, 100),
    "systolic_bp": (None, 120),
    "diastolic_bp": (None, 80),
    "respiratory_rate": (12, 20),
    "blood_oxygen_level": (95, 100),
    "body_temperature": (36, 38),
    "sleep_hours": (7, 9)
}



# Streamlit App
st.set_page_config(page_title="Skavch Digital Health Engine", page_icon="📊", layout="wide")

# Add an image to the header
st.image("bg1.jpg", use_column_width=True)

st.title("Skavch Digital Health Engine")

# Input form for user
st.header("Input Your Health Parameters")
heart_rate = st.number_input("Heart Rate")
systolic_bp = st.number_input("Systolic Blood Pressure")
diastolic_bp = st.number_input("Diastolic Blood Pressure")
respiratory_rate = st.number_input("Respiratory Rate")
blood_oxygen_level = st.number_input("Blood Oxygen Level")
body_temperature = st.number_input("Body Temperature")
sleep_hours = st.number_input("Sleep Hours")
activity_minutes = st.number_input("Activity Minutes")
blood_glucose = st.number_input("Blood Glucose")
ecg_score = st.number_input("ECG Score")

# Classify button
if st.button("Classify"):
    # Preparing input data for prediction
    input_data = np.array([[heart_rate, systolic_bp, diastolic_bp, respiratory_rate, 
                            blood_oxygen_level, body_temperature, sleep_hours, 
                            activity_minutes, blood_glucose, ecg_score]])
    input_data = scaler.transform(input_data)  # Scale input data
    
    # Prediction
    prediction = model.predict(input_data)
    prediction_label = {0: 'healthy_state', 1: 'immediate_consultation', 2: 'immediate_admission'}
    predicted_state = prediction_label[prediction[0]]
    
    # Evaluation metrics on test data
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # Display metrics
    st.write(f"**Model Evaluation Metrics:**")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Get predicted probabilities for the user input
    user_prob = model.predict_proba(input_data)[0]

    # Plot ROC curve regions
    plt.figure(figsize=(8, 6))
    
    # Define regions
    plt.axvspan(0.0, 0.5, color='lightgreen', alpha=0.3)
    plt.axvspan(0.51, 0.79, color='lightyellow', alpha=0.3)
    plt.axvspan(0.80, 1.0, color='lightcoral', alpha=0.3)

    # Add labels for regions
    plt.text(0.25, 0.5, 'Healthy state', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightgreen', alpha=0.5), rotation=90)
    plt.text(0.65, 0.5, 'Immediate Consultation', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightyellow', alpha=0.5), rotation=90)
    plt.text(0.90, 0.5, 'Immediate Admission', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightcoral', alpha=0.5), rotation=90)

    # Scatter the user's predicted probability on the plot
    failure_prob = user_prob[2]  # Assuming '2' corresponds to the failure class
    plt.scatter(failure_prob, failure_prob, color='blue', marker='o', s=100, label=f'User Input Prob = {failure_prob:.2f}')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Condition')
    plt.ylabel('Probabilities')
    plt.title('Digital Health Assessor')
    plt.legend(loc='upper left')

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Display prediction with colored border and background fill around the predicted state
    if predicted_state == 'healthy_state':
        st.markdown(
            f"<div style='border: 2px solid lightgreen; background-color: lightgreen; padding: 10px; display: inline-block;'>"
            f"<h3>Predicted Health State: **{predicted_state}**</h3>"
            f"</div>", unsafe_allow_html=True)
    elif predicted_state == 'immediate_consultation':
        st.markdown(
            f"<div style='border: 2px solid lightyellow; background-color: lightyellow; padding: 10px; display: inline-block;'>"
            f"<h3>Predicted Health State: **{predicted_state}**</h3>"
            f"</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='border: 2px solid lightcoral; background-color: lightcoral; padding: 10px; display: inline-block;'>"
            f"<h3>Predicted Health State: **{predicted_state}**</h3>"
            f"</div>", unsafe_allow_html=True)

    # --- Ideal Range Checks ---
    st.subheader("Ideal Range Check")
    warnings = []

    # Define the user inputs in a dictionary to check each parameter
    user_inputs = {
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "respiratory_rate": respiratory_rate,
        "blood_oxygen_level": blood_oxygen_level,
        "body_temperature": body_temperature,
        "sleep_hours": sleep_hours
    }

    for param, value in user_inputs.items():
        low, high = ideal_ranges[param]
        if low is not None and value < low:
            warnings.append(f"{param.replace('_', ' ').title()}: Below ideal range (Ideal: {low}-{high})")
        elif high is not None and value > high:
            warnings.append(f"{param.replace('_', ' ').title()}: Above ideal range (Ideal: {low}-{high})")

    if warnings:
        st.warning("Some values are out of the ideal range:")
        for warning in warnings:
            st.write(warning)
    else:
        st.success("All values are within the ideal range!")

    # Note on Ideal Ranges
    st.write("**Ideal Ranges for Health Parameters:**")
    st.write("""
    - Heart Rate: 60-100 bpm
    - Systolic Blood Pressure: Less than 120 mmHg
    - Diastolic Blood Pressure: Less than 80 mmHg
    - Respiratory Rate: 12-20 breaths per minute
    - Blood Oxygen Level: 95-100%
    - Body Temperature: 36-38°C
    - Sleep Hours: 7-9 hours
    """)
