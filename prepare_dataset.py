import pandas as pd

# Load dataset
df = pd.read_csv("dataset/cleaned_nutrition_dataset_per100g.csv")

# Select important columns
selected_columns = [
    'food',
    'Sugars (g per 100g)',
    'Carbohydrates (g per 100g)',
    'Calories (kcal per 100g)',
    'Dietary Fiber (g per 100g)',
    'Protein (g per 100g)',
    'Fat (g per 100g)',
    'Sodium (mg per 100g)'
]

df = df[selected_columns]

# Rename columns
df.columns = [
    'Food',
    'Sugar',
    'Carbs',
    'Calories',
    'Fiber',
    'Protein',
    'Fat',
    'Sodium'
]

# Create labels
def classify_food(row):

    if row['Sugar'] > 20:
        return "Avoid"

    elif row['Fiber'] > 5 and row['Sugar'] < 10:
        return "Highly Recommended"

    elif row['Carbs'] > 30:
        return "Moderate"

    else:
        return "Recommended"

# Apply labels
df['Label'] = df.apply(classify_food, axis=1)

# Save new dataset
df.to_csv("dataset/diabetes_ready_dataset.csv", index=False)

print("✅ Diabetes dataset prepared successfully!")
print(df.head())