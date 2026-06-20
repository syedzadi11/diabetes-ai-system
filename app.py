import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load dataset
df = pd.read_csv("dataset/diabetes_ready_dataset.csv")

# -----------------------------
# TITLE
# -----------------------------

st.title("🍽 Diabetes Food Recommendation System")

# -----------------------------
# USER INFORMATION
# -----------------------------

st.subheader("Patient Information")

age = st.number_input("Age", min_value=1, max_value=100)

weight = st.number_input("Weight (kg)", min_value=1)

bmi = st.number_input("BMI", min_value=1.0)

glucose = st.number_input("Current Glucose Level", min_value=50)

diabetes_type = st.selectbox(
    "Diabetes Type",
    ["Type 1", "Type 2"]
)

activity = st.selectbox(
    "Activity Level",
    ["Low", "Moderate", "High"]
)

# -----------------------------
# FOOD INPUT
# -----------------------------

# food = st.text_input("Enter food name")
food = st.selectbox(
    "Select Food",
    sorted(df["Food"].unique())
)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------

if st.button("Predict"):

    # Search food
    row = df[df['Food'].str.lower() == food.lower()]

    if not row.empty:

        # Extract features
        sugar = row['Sugar'].values[0]
        carbs = row['Carbs'].values[0]
        calories = row['Calories'].values[0]
        fiber = row['Fiber'].values[0]
        protein = row['Protein'].values[0]
        fat = row['Fat'].values[0]
        sodium = row['Sodium'].values[0]

        # ML input
        sample = [[
            sugar,
            carbs,
            calories,
            fiber,
            protein,
            fat,
            sodium
        ]]

        # Predict
        prediction = model.predict(sample)

        # Decode label
        result = encoder.inverse_transform(prediction)

        # -----------------------------
        # PERSONALIZED LOGIC
        # -----------------------------

        final_result = result[0]

        # High glucose + high sugar
        if glucose > 180 and sugar > 15:
            final_result = "Avoid"

        # High BMI + high carbs
        elif bmi > 30 and carbs > 25:
            final_result = "Moderate"

        # Active users
        elif activity == "High" and final_result == "Moderate":
            final_result = "Recommended"

        # -----------------------------
        # SHOW RESULT
        # -----------------------------

        st.success(f"Prediction: {final_result}")

        # -----------------------------
        # EXPLAINABILITY
        # -----------------------------

        reasons = []

        if sugar > 20:
            reasons.append("High sugar")

        if carbs > 30:
            reasons.append("High carbohydrates")

        if calories > 300:
            reasons.append("High calories")

        if fiber < 2:
            reasons.append("Low fiber")

        if sodium > 500:
            reasons.append("High sodium")

        # Show reasons
        if reasons:

            st.subheader("Reason")

            for reason in reasons:
                st.write(f"- {reason}")

    else:
        st.error("Food not found in dataset")