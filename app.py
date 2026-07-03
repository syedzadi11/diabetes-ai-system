


# # =========================================================================================

# import streamlit as st
# import pandas as pd
# import pickle

# from personalization import personalize_recommendation, get_nutrient_reasons
# from image_recognition import predict_food_from_image, match_food_to_dataset

# # -----------------------------
# # PAGE CONFIG (must be the first Streamlit call)
# # -----------------------------
# st.set_page_config(
#     page_title="Diabetes Food Advisor",
#     page_icon="🍽",
#     layout="wide",
# )

# # -----------------------------
# # LOAD MODEL + DATA
# # -----------------------------


# @st.cache_resource(show_spinner=False)
# def load_model_and_encoder():
#     with open("model.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("label_encoder.pkl", "rb") as f:
#         encoder = pickle.load(f)
#     return model, encoder


# @st.cache_data(show_spinner=False)
# def load_dataset():
#     return pd.read_csv("dataset/diabetes_ready_dataset.csv")


# model, encoder = load_model_and_encoder()
# df = load_dataset()

# FEATURE_COLUMNS = ['Sugar', 'Carbs', 'Calories', 'Fiber', 'Protein', 'Fat', 'Sodium']

# # Visual styling for each recommendation label: background color, text
# # color, and an emoji so the result is instantly readable at a glance
# # instead of requiring the user to read text carefully.
# LABEL_STYLE = {
#     "Highly Recommended": {"color": "#1e7e34", "bg": "#d4edda", "emoji": "✅"},
#     "Recommended":        {"color": "#0c5460", "bg": "#d1ecf1", "emoji": "🙂"},
#     "Moderate":            {"color": "#856404", "bg": "#fff3cd", "emoji": "⚠️"},
#     "Avoid":               {"color": "#721c24", "bg": "#f8d7da", "emoji": "🚫"},
# }


# def render_result_card(label):
#     style = LABEL_STYLE.get(label, {"color": "#000", "bg": "#eee", "emoji": ""})
#     st.markdown(
#         f"""
#         <div style="
#             background-color:{style['bg']};
#             color:{style['color']};
#             padding: 20px;
#             border-radius: 12px;
#             text-align: center;
#             font-size: 28px;
#             font-weight: 700;
#             margin-bottom: 10px;
#         ">
#             {style['emoji']} {label}
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


# # -----------------------------
# # SIDEBAR: PATIENT INFORMATION
# # -----------------------------
# # Moving patient info into the sidebar keeps the main page focused on
# # the actual task (choosing food + seeing the result), rather than
# # making the user scroll past a long form every time.

# with st.sidebar:
#     st.header("👤 Patient Information")

#     age = st.number_input("Age", min_value=1, max_value=100, value=30)
#     weight = st.number_input("Weight (kg)", min_value=1, value=70)
#     height_cm = st.number_input("Height (cm)", min_value=50, value=170)

#     # IMPROVEMENT: BMI is calculated automatically from height + weight
#     # instead of asking the user to compute and type it in themselves
#     # (most people don't know their BMI off the top of their head, and
#     # asking for it directly was a usability dead-end).
#     height_m = height_cm / 100
#     bmi = round(weight / (height_m ** 2), 1) if height_m > 0 else 0
#     st.metric("Calculated BMI", bmi)

#     glucose = st.number_input("Current Glucose Level (mg/dL)", min_value=50, value=100)

#     diabetes_type = st.selectbox("Diabetes Type", ["Type 1", "Type 2"])
#     activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])

#     st.divider()
#     st.caption(
#         "This tool gives general guidance based on nutrition data and is "
#         "not a substitute for medical advice from your doctor."
#     )

# # -----------------------------
# # MAIN AREA
# # -----------------------------

# st.title("🍽 Diabetes Food Advisor")
# st.caption("Find out if a food fits your diabetic diet -- by name or by photo.")

# st.markdown("### Step 1 -- Choose your food")

# tab_manual, tab_image = st.tabs(["📋 Select from List", "📷 Upload Photo"])

# with tab_manual:
#     food = st.selectbox("Search or select a food", sorted(df["Food"].unique()))
#     if st.button("Check this food", type="primary", key="manual_select_btn"):
#         match = df[df['Food'].str.lower() == food.lower()]
#         if not match.empty:
#             st.session_state["selected_food_row"] = match.iloc[[0]]
#             st.session_state["selected_food_image"] = None

# with tab_image:
#     st.info(
#         "📌 Best for common prepared dishes (pizza, burger, fries, sushi, "
#         "salad, etc). Not designed for plain raw ingredients like a "
#         "single vegetable."
#     )
#     uploaded_image = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])

#     if uploaded_image is not None:
#         img_col, btn_col = st.columns([1, 2])
#         with img_col:
#             st.image(uploaded_image, caption="Your photo", width=220)

#         with btn_col:
#             if st.button("🔍 Identify Food", type="primary", key="identify_food_btn"):
#                 with st.spinner("Analyzing image... this can take a few seconds the first time."):
#                     try:
#                         predictions = predict_food_from_image(uploaded_image, top_k=3)
#                     except Exception:
#                         st.error("Could not analyze this image. Please try a different photo.")
#                         predictions = []

#                 if predictions:
#                     top_prediction = predictions[0]
#                     confidence = top_prediction["score"] * 100
#                     label_title = top_prediction["label"].replace("_", " ").title()

#                     st.success(f"**Best guess:** {label_title}  ({confidence:.1f}% confidence)")

#                     with st.expander("See other possibilities"):
#                         for p in predictions[1:]:
#                             st.write(f"- {p['label'].replace('_', ' ').title()} ({p['score']*100:.1f}%)")

#                     matched_row, suggestions = match_food_to_dataset(top_prediction["label"], df)

#                     if matched_row is not None:
#                         st.session_state["selected_food_row"] = matched_row
#                         st.session_state["selected_food_image"] = uploaded_image
#                         st.success(f"Matched to our database: **{matched_row['Food'].values[0]}**")
#                     elif suggestions:
#                         st.warning("No exact match found. Did you mean one of these?")
#                         chosen = st.radio("Closest matches:", suggestions, key="suggestion_radio")
#                         if st.button("Use this match", key="use_suggestion_btn"):
#                             match = df[df['Food'] == chosen]
#                             st.session_state["selected_food_row"] = match.iloc[[0]]
#                             st.session_state["selected_food_image"] = uploaded_image
#                     else:
#                         st.error(
#                             "This food isn't in our database yet. "
#                             "Please use the 'Select from List' tab instead."
#                         )

# # -----------------------------
# # RESULT SECTION
# # -----------------------------

# selected_food_row = st.session_state.get("selected_food_row")
# selected_food_image = st.session_state.get("selected_food_image")

# if selected_food_row is not None:

#     st.divider()
#     st.markdown("### Step 2 -- Result")

#     food_name = selected_food_row['Food'].values[0]
#     sugar = selected_food_row['Sugar'].values[0]
#     carbs = selected_food_row['Carbs'].values[0]
#     calories = selected_food_row['Calories'].values[0]
#     fiber = selected_food_row['Fiber'].values[0]
#     protein = selected_food_row['Protein'].values[0]
#     fat = selected_food_row['Fat'].values[0]
#     sodium = selected_food_row['Sodium'].values[0]

#     sample = pd.DataFrame(
#         [[sugar, carbs, calories, fiber, protein, fat, sodium]],
#         columns=FEATURE_COLUMNS,
#     )
#     prediction = model.predict(sample)
#     base_label = encoder.inverse_transform(prediction)[0]

#     final_result, personalization_notes = personalize_recommendation(
#         base_label=base_label,
#         sugar=sugar,
#         carbs=carbs,
#         glucose=glucose,
#         bmi=bmi,
#         activity=activity,
#         diabetes_type=diabetes_type,
#     )

#     left_col, right_col = st.columns([1, 2])

#     with left_col:
#         if selected_food_image is not None:
#             st.image(selected_food_image, caption=food_name, use_container_width=True)
#         else:
#             st.markdown(f"#### {food_name}")
#             st.caption("Selected from list (no photo)")

#     with right_col:
#         render_result_card(final_result)

#         if base_label != final_result:
#             st.caption(f"ℹ️ Base model prediction was '{base_label}', adjusted for your personal profile.")

#         # Nutrient values shown as compact metric tiles instead of a
#         # plain wall of text -- easier to scan at a glance.
#         m1, m2, m3, m4 = st.columns(4)
#         m1.metric("Sugar (g)", sugar)
#         m2.metric("Carbs (g)", carbs)
#         m3.metric("Calories", calories)
#         m4.metric("Fiber (g)", fiber)

#     if personalization_notes:
#         st.markdown("#### 🧑‍⚕️ Personalized Notes")
#         for note in personalization_notes:
#             st.write(f"- {note}")

#     reasons = get_nutrient_reasons(sugar, carbs, calories, fiber, sodium)
#     if reasons:
#         st.markdown("#### 📋 Why this result")
#         for reason in reasons:
#             st.write(f"- {reason}")
# else:
#     st.info("👆 Choose a food above (from the list or by photo) to see your result.")







# ================================================================================

import streamlit as st
import pandas as pd
import pickle

from personalization import personalize_recommendation, get_nutrient_reasons
from image_recognition import predict_food_from_image, match_food_to_dataset

# -----------------------------
# PAGE CONFIG (must be the first Streamlit call)
# -----------------------------
st.set_page_config(
    page_title="Diabetes Food Advisor",
    page_icon="🍽",
    layout="wide",
)

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------


@st.cache_resource(show_spinner=False)
def load_model_and_encoder():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv("dataset/diabetes_ready_dataset.csv")


model, encoder = load_model_and_encoder()
df = load_dataset()

FEATURE_COLUMNS = ['Sugar', 'Carbs', 'Calories', 'Fiber', 'Protein', 'Fat', 'Sodium']

# Visual styling for each recommendation label: background color, text
# color, and an emoji so the result is instantly readable at a glance
# instead of requiring the user to read text carefully.
LABEL_STYLE = {
    "Highly Recommended": {"color": "#1e7e34", "bg": "#d4edda", "emoji": "✅"},
    "Recommended":        {"color": "#0c5460", "bg": "#d1ecf1", "emoji": "🙂"},
    "Moderate":            {"color": "#856404", "bg": "#fff3cd", "emoji": "⚠️"},
    "Avoid":               {"color": "#721c24", "bg": "#f8d7da", "emoji": "🚫"},
}


def render_result_card(label):
    style = LABEL_STYLE.get(label, {"color": "#000", "bg": "#eee", "emoji": ""})
    st.markdown(
        f"""
        <div style="
            background-color:{style['bg']};
            color:{style['color']};
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
        ">
            {style['emoji']} {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# SIDEBAR: PATIENT INFORMATION
# -----------------------------
# Moving patient info into the sidebar keeps the main page focused on
# the actual task (choosing food + seeing the result), rather than
# making the user scroll past a long form every time.

with st.sidebar:
    st.header("👤 Patient Information")

    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    weight = st.number_input("Weight (kg)", min_value=1, value=70)
    height_cm = st.number_input("Height (cm)", min_value=50, value=170)

    # IMPROVEMENT: BMI is calculated automatically from height + weight
    # instead of asking the user to compute and type it in themselves
    # (most people don't know their BMI off the top of their head, and
    # asking for it directly was a usability dead-end).
    height_m = height_cm / 100
    bmi = round(weight / (height_m ** 2), 1) if height_m > 0 else 0
    st.metric("Calculated BMI", bmi)

    glucose = st.number_input("Current Glucose Level (mg/dL)", min_value=50, value=100)

    diabetes_type = st.selectbox("Diabetes Type", ["Type 1", "Type 2"])
    activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])

    st.divider()
    st.caption(
        "This tool gives general guidance based on nutrition data and is "
        "not a substitute for medical advice from your doctor."
    )

# -----------------------------
# MAIN AREA
# -----------------------------

st.title("🍽 Diabetes Food Advisor")
st.caption("Find out if a food fits your diabetic diet -- by name or by photo.")

st.markdown("### Step 1 -- Choose your food")

tab_manual, tab_image = st.tabs(["📋 Select from List", "📷 Upload Photo"])

with tab_manual:
    food = st.selectbox("Search or select a food", sorted(df["Food"].unique()))
    if st.button("Check this food", type="primary", key="manual_select_btn"):
        match = df[df['Food'].str.lower() == food.lower()]
        if not match.empty:
            st.session_state["selected_food_row"] = match.iloc[[0]]
            st.session_state["selected_food_image"] = None

with tab_image:
    st.info(
        "📌 Best for common prepared dishes (pizza, burger, fries, sushi, "
        "salad, etc). Not designed for plain raw ingredients like a "
        "single vegetable."
    )
    uploaded_image = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        img_col, btn_col = st.columns([1, 2])
        with img_col:
            st.image(uploaded_image, caption="Your photo", width=220)

        with btn_col:
            if st.button("🔍 Identify Food", type="primary", key="identify_food_btn"):
                with st.spinner("Analyzing image... this can take a few seconds the first time."):
                    try:
                        predictions = predict_food_from_image(uploaded_image, top_k=3)
                    except Exception:
                        st.error("Could not analyze this image. Please try a different photo.")
                        predictions = []

                if predictions:
                    top_prediction = predictions[0]
                    confidence = top_prediction["score"] * 100
                    label_title = top_prediction["label"].replace("_", " ").title()

                    st.success(f"**Best guess:** {label_title}  ({confidence:.1f}% confidence)")

                    with st.expander("See other possibilities"):
                        for p in predictions[1:]:
                            st.write(f"- {p['label'].replace('_', ' ').title()} ({p['score']*100:.1f}%)")

                    # FIX: previously only the top-1 prediction was
                    # matched against the dataset, so if the top guess
                    # wasn't in the database (e.g. model said "Paella"
                    # for a Biryani photo, and "paella" isn't in the
                    # dataset), the user immediately saw "not found".
                    # Now we try ALL top-3 predictions in order, and use
                    # the first one that matches -- so "Fried Rice"
                    # (the 2nd guess) can still save the result even
                    # when "Paella" (1st guess) has no match.
                    matched_row = None
                    suggestions = []
                    matched_via = None

                    for pred in predictions:
                        row, sugg = match_food_to_dataset(pred["label"], df)
                        if row is not None:
                            matched_row = row
                            matched_via = pred["label"].replace("_", " ").title()
                            break
                        if sugg and not suggestions:
                            suggestions = sugg

                    if matched_row is not None:
                        st.session_state["selected_food_row"] = matched_row
                        st.session_state["selected_food_image"] = uploaded_image
                        matched_name = matched_row['Food'].values[0]
                        if matched_via and matched_via.lower() != label_title.lower():
                            st.info(
                                f"Top guess '{label_title}' wasn't in our database, "
                                f"but we matched via '{matched_via}' → **{matched_name}**"
                            )
                        else:
                            st.success(f"Matched to our database: **{matched_name}**")
                    elif suggestions:
                        st.warning("No exact match found. Did you mean one of these?")
                        chosen = st.radio("Closest matches:", suggestions, key="suggestion_radio")
                        if st.button("Use this match", key="use_suggestion_btn"):
                            match = df[df['Food'] == chosen]
                            st.session_state["selected_food_row"] = match.iloc[[0]]
                            st.session_state["selected_food_image"] = uploaded_image
                    else:
                        st.error(
                            "This food isn't in our database yet. "
                            "Please use the 'Select from List' tab instead."
                        )

# -----------------------------
# RESULT SECTION
# -----------------------------

selected_food_row = st.session_state.get("selected_food_row")
selected_food_image = st.session_state.get("selected_food_image")

if selected_food_row is not None:

    st.divider()
    st.markdown("### Step 2 -- Result")

    food_name = selected_food_row['Food'].values[0]
    sugar = selected_food_row['Sugar'].values[0]
    carbs = selected_food_row['Carbs'].values[0]
    calories = selected_food_row['Calories'].values[0]
    fiber = selected_food_row['Fiber'].values[0]
    protein = selected_food_row['Protein'].values[0]
    fat = selected_food_row['Fat'].values[0]
    sodium = selected_food_row['Sodium'].values[0]

    sample = pd.DataFrame(
        [[sugar, carbs, calories, fiber, protein, fat, sodium]],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(sample)
    base_label = encoder.inverse_transform(prediction)[0]

    final_result, personalization_notes = personalize_recommendation(
        base_label=base_label,
        sugar=sugar,
        carbs=carbs,
        glucose=glucose,
        bmi=bmi,
        activity=activity,
        diabetes_type=diabetes_type,
    )

    left_col, right_col = st.columns([1, 2])

    with left_col:
        if selected_food_image is not None:
            st.image(selected_food_image, caption=food_name, use_container_width=True)
        else:
            st.markdown(f"#### {food_name}")
            st.caption("Selected from list (no photo)")

    with right_col:
        render_result_card(final_result)

        if base_label != final_result:
            st.caption(f"ℹ️ Base model prediction was '{base_label}', adjusted for your personal profile.")

        # Nutrient values shown as compact metric tiles instead of a
        # plain wall of text -- easier to scan at a glance.
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sugar (g)", sugar)
        m2.metric("Carbs (g)", carbs)
        m3.metric("Calories", calories)
        m4.metric("Fiber (g)", fiber)

    if personalization_notes:
        st.markdown("#### 🧑‍⚕️ Personalized Notes")
        for note in personalization_notes:
            st.write(f"- {note}")

    reasons = get_nutrient_reasons(sugar, carbs, calories, fiber, sodium)
    if reasons:
        st.markdown("#### 📋 Why this result")
        for reason in reasons:
            st.write(f"- {reason}")
else:
    st.info("👆 Choose a food above (from the list or by photo) to see your result.")