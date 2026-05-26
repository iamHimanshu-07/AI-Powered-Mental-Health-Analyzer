import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io

import os
print(os.path.exists("../notebooks/mental_health_model.pkl"))
print(os.getcwd())


# Load model and MultiLabelBinarizer
pipeline = joblib.load("../notebooks/mental_health_model.pkl")
mlb = joblib.load("../notebooks/mlb.pkl")

# Severity color mapping for each emotion
severity_colors = {
    'anger': 'orange',
    'sadness': 'blue',
    'emptiness': 'gray',
    'hopelessness': 'darkred',
    'worthlessness': 'purple',
    'loneliness': 'lightblue',
    'suicide intent': 'red',
    'brain dysfunction (forget)': 'green'
}

# UI
st.title("🧠 Mental Health Emotion Detector")
st.write("Enter a text to detect emotional signals:")

text_input = st.text_area("Your text here")

# Add threshold slider
threshold = st.slider("Prediction Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        vec = [text_input]
        proba = pipeline.predict_proba(vec)[0]
        prediction = (proba >= threshold).astype(int)

        try:
            predicted_labels = mlb.inverse_transform(np.array([prediction]))[0]
        except Exception as e:
            st.error(f"Error in decoding labels: {e}")
            predicted_labels = []

        st.subheader("🔍 Predicted Emotions:")
        if predicted_labels:
            st.success(", ".join(predicted_labels))

            # Risk flag warning
            high_risk_labels = {"suicide intent", "hopelessness"}
            if any(label in predicted_labels for label in high_risk_labels):
                st.warning("⚠️ High-risk emotional indicators detected. If this is real data, please consider seeking professional help.")
        else:
            st.info("No strong emotions detected with the current threshold.")

        # Bar chart visualization
        emotions = mlb.classes_
        colors = [severity_colors.get(label, 'gray') for label in emotions]
        fig, ax = plt.subplots()
        ax.barh(emotions, proba, color=colors)
        ax.set_xlabel("Probability")
        ax.set_title("Emotion Prediction Probabilities")
        st.pyplot(fig)

        # ✅ Fixed download button
        result_text = io.StringIO()
        result_text.write("Predicted Emotions:\n")
        result_text.write(", ".join(predicted_labels) + "\n\n")
        result_text.write("Emotion Probabilities:\n")
        for emo, prob in zip(emotions, proba):
            result_text.write(f"{emo}: {prob:.2f}\n")

        st.download_button(
            label="📄 Download Results as TXT",
            data=result_text.getvalue(),
            file_name="mental_health_predictions.txt",
            mime="text/plain"
        )
