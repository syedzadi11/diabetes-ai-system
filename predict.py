import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load dataset
df = pd.read_csv("dataset/diabetes_ready_dataset.csv")

# User input
food = input("Enter food name: ").lower()

# Search food
row = df[df['Food'].str.lower() == food]

if not row.empty:

    # Extract values
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

    # Prediction
    prediction = model.predict(sample)

    # Decode label
    result = encoder.inverse_transform(prediction)

    # Show result
    print(f"\n🍽 Food: {food}")
    print(f"✅ Prediction: {result[0]}")

    # -----------------------------
    # Explainability Logic
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
        print("\n🧠 Reason:")
        for reason in reasons:
            print(f"- {reason}")

else:
    print("❗ Food not found in dataset")