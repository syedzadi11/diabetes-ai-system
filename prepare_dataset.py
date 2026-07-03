# import pandas as pd

# # Load dataset
# df = pd.read_csv("dataset/cleaned_nutrition_dataset_per100g.csv")

# # Select important columns
# selected_columns = [
#     'food',
#     'Sugars (g per 100g)',
#     'Carbohydrates (g per 100g)',
#     'Calories (kcal per 100g)',
#     'Dietary Fiber (g per 100g)',
#     'Protein (g per 100g)',
#     'Fat (g per 100g)',
#     'Sodium (mg per 100g)'
# ]

# df = df[selected_columns]

# # Rename columns
# df.columns = [
#     'Food',
#     'Sugar',
#     'Carbs',
#     'Calories',
#     'Fiber',
#     'Protein',
#     'Fat',
#     'Sodium'
# ]

# # Create labels
# def classify_food(row):

#     if row['Sugar'] > 20:
#         return "Avoid"

#     elif row['Fiber'] > 5 and row['Sugar'] < 10:
#         return "Highly Recommended"

#     elif row['Carbs'] > 30:
#         return "Moderate"

#     else:
#         return "Recommended"

# # Apply labels
# df['Label'] = df.apply(classify_food, axis=1)

# # Save new dataset
# df.to_csv("dataset/diabetes_ready_dataset.csv", index=False)

# print("✅ Diabetes dataset prepared successfully!")
# print(df.head())







import pandas as pd

# -----------------------------
# LOAD DATASET
# -----------------------------

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

# -----------------------------
# WHY THE OLD LOGIC WAS WRONG
# -----------------------------
# Old rule:
#   Sugar > 20            -> Avoid
#   Fiber > 5 & Sugar < 10 -> Highly Recommended   <-- ignored Calories/Fat entirely
#   Carbs > 30             -> Moderate
#   else                   -> Recommended
#
# Problem: a 1000-kcal, 65g-fat snack with low sugar and decent fiber
# (e.g. fried chips) could become "Highly Recommended". Calories and Fat
# were collected but never used to decide the label.
#
# New approach: build a single "diabetes risk score" using ALL relevant
# nutrients, each weighted by how much it actually matters for blood sugar
# and metabolic health. Lower score = better for a diabetic diet.

def diabetes_risk_score(row):
    """
    Returns a numeric risk score (higher = worse for diabetes).

    Key ideas baked into this score:
    1. Blood-sugar impact (glycemic effect) should dominate the score.
    2. A zero-carb, zero-sugar item (oil, plain spice, pure protein)
       should NOT be flagged as "Avoid" just because it is
       calorie-dense -- fats/oils in moderation don't spike blood
       glucose the way sugar/refined carbs do.
    3. Sugar from WHOLE foods that also carry fiber (fruits, some
       vegetables) is absorbed more slowly than sugar from refined/
       processed foods (soda, cake, candy, fruit juice) with little/no
       fiber. Medical guidance (ADA / Mayo Clinic) generally places
       whole fruits like mango/banana/apple in a "eat in moderation"
       category, NOT an "avoid" category -- "avoid" is reserved for
       refined sugar sources (soda, candy, juice, pastries) that have
       no fiber to slow absorption. We apply a strong "natural sugar
       discount" based on how much fiber accompanies the sugar to
       reflect this.
    """
    sugar = row['Sugar']
    carbs = row['Carbs']
    calories = row['Calories']
    fiber = row['Fiber']
    fat = row['Fat']
    sodium = row['Sodium']

    # Fiber-to-sugar ratio approximates "how whole/unprocessed" the sugar
    # source is. Capped so very high-fiber low-sugar foods don't make the
    # discount explode.
    fiber_ratio = min(fiber / sugar, 1.0) if sugar > 0 else 0
    # Discount: up to 65% off the sugar penalty when fiber is abundant
    # relative to sugar (typical of whole fruit/veg), 0% off when there's
    # no fiber at all (typical of refined sugar/soda/candy/juice).
    sugar_multiplier = 3.0 * (1 - 0.65 * fiber_ratio)

    score = 0.0

    # Sugar: the single biggest driver of blood-glucose spikes (with the
    # natural-food discount applied above).
    score += sugar * sugar_multiplier

    # Net carbs (carbs minus fiber) approximate the carbs that actually
    # get digested and raise blood sugar -- the standard "net carb" idea
    # used in diabetic diet planning.
    net_carbs = max(carbs - fiber, 0)
    score += net_carbs * 1.0

    # Calories: only a SECONDARY/minor signal, and only really matters
    # once a food also carries meaningful carbs (i.e. it's a carb-heavy
    # AND calorie-dense food, like fried snacks or desserts -- not a
    # plain oil or spice). Scale the calorie penalty by how carb-heavy
    # the food is, instead of applying it unconditionally.
    carb_fraction = min(carbs / 100.0, 1.0)  # 0 (no carbs) -> 1 (all carbs)
    score += calories * 0.015 * carb_fraction

    # Fat: very small standalone weight. Pure fats/oils are not a direct
    # blood-sugar risk, so fat alone shouldn't push something to "Avoid".
    # It still nudges ultra-processed combo foods (fried + sugary) up a
    # bit.
    score += fat * 0.05

    # Sodium: not glycemic, but relevant for cardiovascular risk that
    # commonly accompanies diabetes. Very light weight (sodium is in mg).
    score += sodium * 0.01

    # Fiber REDUCES risk -- it slows glucose absorption and is the reason
    # whole grains/legumes/vegetables are diabetic-friendly.
    score -= fiber * 4.0

    return score


def classify_food(row):
    score = diabetes_risk_score(row)

    # Thresholds were tuned against known reference points:
    #   Highly Recommended: non-starchy veg, legumes, lean protein
    #   Recommended:        whole grains, nuts, plain dairy
    #   Moderate:           whole fruits in moderation (mango, banana,
    #                        apple, watermelon), starchy staples
    #   Avoid:               refined sugar, soda, candy, fruit juice,
    #                        pastries, fried/sugary combo foods
    if score <= 2:
        return "Highly Recommended"
    elif score <= 8:
        return "Recommended"
    elif score <= 45:
        return "Moderate"
    else:
        return "Avoid"


# -----------------------------
# MANUAL OVERRIDES
# -----------------------------
# A handful of foods in the source dataset have nutrient values that are
# known to be wrong/misleading for our purposes (e.g. artificial
# sweeteners listed with a non-zero "Sugar" value, even though they are
# sugar substitutes with no real glycemic impact). Rather than trying to
# fix the entire raw dataset, we explicitly override the label for these
# specific, well-known cases so they don't get an incorrect score purely
# from bad source data.
MANUAL_OVERRIDES = {
    # Artificial/non-nutritive sweeteners: no real sugar, safe for
    # diabetics in normal amounts, despite the dataset listing a "Sugar"
    # value for them.
    "splenda sweetener": "Highly Recommended",
    "aspartame sweetener": "Highly Recommended",
    "saccharin sweetener": "Highly Recommended",
}


def apply_manual_overrides(df):
    for food_name, forced_label in MANUAL_OVERRIDES.items():
        mask = df['Food'].str.lower() == food_name.lower()
        df.loc[mask, 'Label'] = forced_label
    return df


# Apply scoring + labels
df['Risk_Score'] = df.apply(diabetes_risk_score, axis=1)
df['Label'] = df.apply(classify_food, axis=1)
df = apply_manual_overrides(df)

# Save new dataset (keep Risk_Score for transparency/debugging, drop later
# if you don't want it in the final ML training data)
df.to_csv("dataset/diabetes_ready_dataset.csv", index=False)

print("Diabetes dataset prepared successfully!")
print(df.head())
print("\nLabel distribution:")
print(df['Label'].value_counts())