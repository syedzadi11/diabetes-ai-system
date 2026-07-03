# import pandas as pd
# import pickle

# # Load model
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# # Load encoder
# with open("label_encoder.pkl", "rb") as f:
#     encoder = pickle.load(f)

# # Load dataset
# df = pd.read_csv("dataset/diabetes_ready_dataset.csv")

# # User input
# food = input("Enter food name: ").lower()

# # Search food
# row = df[df['Food'].str.lower() == food]

# if not row.empty:

#     # Extract values
#     sugar = row['Sugar'].values[0]
#     carbs = row['Carbs'].values[0]
#     calories = row['Calories'].values[0]
#     fiber = row['Fiber'].values[0]
#     protein = row['Protein'].values[0]
#     fat = row['Fat'].values[0]
#     sodium = row['Sodium'].values[0]

#     # ML input
#     sample = [[
#         sugar,
#         carbs,
#         calories,
#         fiber,
#         protein,
#         fat,
#         sodium
#     ]]

#     # Prediction
#     prediction = model.predict(sample)

#     # Decode label
#     result = encoder.inverse_transform(prediction)

#     # Show result
#     print(f"\n🍽 Food: {food}")
#     print(f"✅ Prediction: {result[0]}")

#     # -----------------------------
#     # Explainability Logic
#     # -----------------------------

#     reasons = []

#     if sugar > 20:
#         reasons.append("High sugar")

#     if carbs > 30:
#         reasons.append("High carbohydrates")

#     if calories > 300:
#         reasons.append("High calories")

#     if fiber < 2:
#         reasons.append("Low fiber")

#     if sodium > 500:
#         reasons.append("High sodium")

#     # Show reasons
#     if reasons:
#         print("\n🧠 Reason:")
#         for reason in reasons:
#             print(f"- {reason}")

# else:
#     print("❗ Food not found in dataset")











import pandas as pd
import pickle

from personalization import personalize_recommendation, get_nutrient_reasons

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

df = pd.read_csv("dataset/diabetes_ready_dataset.csv")

FEATURE_COLUMNS = ['Sugar', 'Carbs', 'Calories', 'Fiber', 'Protein', 'Fat', 'Sodium']


def get_float_input(prompt, default=None, min_value=None):
    """
    Small helper to safely read a numeric value from the CLI.
    FIX: the old predict.py had no error handling at all for user input
    (e.g. food name typos). This helper at least protects the numeric
    inputs added below for personalization, so a bad value doesn't crash
    the whole script.
    """
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = float(raw)
            if min_value is not None and value < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def find_food(df, food_name):
    """
    FIX: the old code only matched on an EXACT lowercase match
    (`df['Food'].str.lower() == food`), so a small typo or partial name
    returned nothing useful. This version falls back to a "contains"
    search and offers suggestions if there's no exact match.
    """
    exact = df[df['Food'].str.lower() == food_name.lower()]
    if not exact.empty:
        return exact, None

    partial = df[df['Food'].str.lower().str.contains(food_name.lower(), na=False)]
    if not partial.empty:
        suggestions = partial['Food'].head(10).tolist()
        return None, suggestions

    return None, []


def main():
    food_input = input("Enter food name: ").strip()

    row, suggestions = find_food(df, food_input)

    if row is None:
        if suggestions:
            print(f"\n'{food_input}' not found exactly. Did you mean one of these?")
            for s in suggestions:
                print(f"  - {s}")
        else:
            print(f"\nFood '{food_input}' not found in dataset.")
        return

    # Extract features
    sugar = row['Sugar'].values[0]
    carbs = row['Carbs'].values[0]
    calories = row['Calories'].values[0]
    fiber = row['Fiber'].values[0]
    protein = row['Protein'].values[0]
    fat = row['Fat'].values[0]
    sodium = row['Sodium'].values[0]
    matched_food_name = row['Food'].values[0]

    sample = pd.DataFrame(
        [[sugar, carbs, calories, fiber, protein, fat, sodium]],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(sample)
    base_label = encoder.inverse_transform(prediction)[0]

    # -----------------------------
    # OPTIONAL PERSONALIZATION (CLI)
    # -----------------------------
    # FIX: previously predict.py had NO personalization at all (it
    # always just printed the raw model label, ignoring glucose/BMI/
    # activity/diabetes type). app.py had this logic but predict.py
    # didn't, so the two tools could give different advice for the same
    # food. We now offer the same optional personalization here, reusing
    # the exact same shared logic as app.py.
    print("\n(Optional) Enter your profile for a personalized result, or press Enter to skip each field.")
    glucose_raw = input("Current glucose level (or Enter to skip): ").strip()

    if glucose_raw == "":
        final_result = base_label
        notes = []
    else:
        glucose = float(glucose_raw)
        bmi = get_float_input("BMI: ", default=22.0, min_value=1.0)
        activity = input("Activity level [Low/Moderate/High] (default Moderate): ").strip() or "Moderate"
        diabetes_type = input("Diabetes type [Type 1/Type 2] (default Type 2): ").strip() or "Type 2"

        final_result, notes = personalize_recommendation(
            base_label=base_label,
            sugar=sugar,
            carbs=carbs,
            glucose=glucose,
            bmi=bmi,
            activity=activity,
            diabetes_type=diabetes_type,
        )

    # -----------------------------
    # SHOW RESULT
    # -----------------------------

    print(f"\nFood: {matched_food_name}")
    print(f"Base model prediction: {base_label}")
    if final_result != base_label:
        print(f"Personalized result: {final_result}")
    for note in notes:
        print(f"  - {note}")

    # -----------------------------
    # EXPLAINABILITY (shared with app.py)
    # -----------------------------

    reasons = get_nutrient_reasons(sugar, carbs, calories, fiber, sodium)

    if reasons:
        print("\nReasons:")
        for reason in reasons:
            print(f"- {reason}")


if __name__ == "__main__":
    main()