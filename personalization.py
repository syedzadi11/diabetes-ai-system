"""
Shared personalization & explainability logic for the Diabetes Food
Recommendation System.

WHY THIS FILE EXISTS:
Previously, the "reasons" list (High sugar / High carbs / etc.) and the
personalization adjustment logic (glucose, BMI, activity overrides) were
copy-pasted in BOTH app.py and predict.py. That meant any fix had to be
made twice, and the two files could silently drift apart. This module is
now the single source of truth -- both app.py and predict.py import from
here.

WHY THE OLD if/elif CHAIN WAS A PROBLEM:
    if glucose > 180 and sugar > 15:      -> "Avoid"
    elif bmi > 30 and carbs > 25:         -> "Moderate"
    elif activity == "High" and ...:      -> "Recommended"

Because this was an if/elif/elif chain, only ONE branch could ever run.
If a person had BOTH high glucose AND high BMI, only the first matching
condition (glucose) would apply -- the BMI risk was silently ignored.

NEW APPROACH:
Each personal factor (glucose, BMI, activity, diabetes type) is evaluated
independently and produces its own "severity vote". We combine all votes
and pick the single most restrictive (worst) outcome. This means multiple
risk factors can each contribute, and none of them get silently dropped
just because an earlier check already matched.
"""

# Severity ranking from best to worst. Used to compare/combine multiple
# recommendations and pick the strictest one.
SEVERITY_ORDER = ["Highly Recommended", "Recommended", "Moderate", "Avoid"]


def _severity_rank(label):
    """Higher number = more restrictive / worse for the patient."""
    if label not in SEVERITY_ORDER:
        return 0
    return SEVERITY_ORDER.index(label)


def _worst(*labels):
    """Given several recommendation labels, return the most restrictive one."""
    valid = [l for l in labels if l in SEVERITY_ORDER]
    if not valid:
        return None
    return max(valid, key=_severity_rank)


def personalize_recommendation(
    base_label,
    sugar,
    carbs,
    glucose,
    bmi,
    activity,
    diabetes_type,
):
    """
    Combine the model's base label with the patient's personal context
    to produce a final recommendation. Every factor below is evaluated
    independently, then the single worst (most restrictive) outcome
    wins -- so no risk factor gets silently ignored just because another
    one matched first.

    Returns: (final_label, list_of_personalization_notes)
    """
    candidate_labels = [base_label]
    notes = []

    # --- Glucose + sugar interaction -----------------------------------
    # Uncontrolled high glucose makes any meaningfully sugary food risky,
    # regardless of what else is going on.
    if glucose > 180 and sugar > 15:
        candidate_labels.append("Avoid")
        notes.append(
            "Your current glucose reading is high and this food has "
            "meaningful sugar content -- treated as higher risk."
        )

    # --- BMI + carbs interaction -----------------------------------
    # Higher BMI combined with a carb-heavy food is a separate concern
    # from glucose, and should NOT be skipped just because the glucose
    # check above already fired.
    if bmi > 30 and carbs > 25:
        candidate_labels.append("Moderate")
        notes.append(
            "Your BMI is in a higher range and this food is carb-heavy -- "
            "portion control is advised."
        )

    # --- Diabetes type adjustment -----------------------------------
    # Type 1 diabetics typically dose insulin around carb intake, so
    # consistent/predictable carbs matter more than the absolute amount.
    # We don't make this swing the label as hard as glucose/BMI, but it
    # nudges a borderline "Recommended" toward "Moderate" if carbs are
    # non-trivial, as a reminder to factor this into insulin dosing.
    if diabetes_type == "Type 1" and carbs > 20 and base_label == "Recommended":
        candidate_labels.append("Moderate")
        notes.append(
            "As a Type 1 diabetic, remember to account for this food's "
            "carbohydrate content in your insulin dosing."
        )

    # --- Activity level adjustment -----------------------------------
    # Higher activity improves insulin sensitivity and glucose
    # utilization, so an otherwise-"Moderate" food can reasonably be
    # treated as acceptable for very active individuals. This only ever
    # RELAXES a "Moderate" result -- it never overrides "Avoid", since
    # activity level doesn't make a genuinely high-risk food safe.
    final_before_activity = _worst(*candidate_labels)
    if activity == "High" and final_before_activity == "Moderate":
        notes.append(
            "Your high activity level improves glucose tolerance, so "
            "this moderate-risk food is more acceptable for you."
        )
        return "Recommended", notes

    final_label = _worst(*candidate_labels)
    return final_label, notes


def get_nutrient_reasons(sugar, carbs, calories, fiber, sodium):
    """
    Returns a list of plain-English reasons explaining why a food might
    be flagged, based on simple per-100g nutrient thresholds. Shared
    between app.py and predict.py so the explanations always stay in
    sync with each other.
    """
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

    return reasons