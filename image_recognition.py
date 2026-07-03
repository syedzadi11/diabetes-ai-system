# """
# Image-based food recognition for the Diabetes Food Recommendation System.

# WHAT THIS DOES:
# Takes a photo of food uploaded by the user and identifies which food it
# is, using a pretrained image classification model (no training required
# from us). The predicted food name is then matched against our nutrition
# dataset (diabetes_ready_dataset.csv) so the existing Random Forest model
# and personalization logic can run on it exactly like a manually-selected
# food would.

# MODEL USED:
# "nateraw/food" -- a Vision Transformer (ViT) fine-tuned on the Food-101
# dataset (101 food categories, ~89% accuracy). This is a publicly
# available pretrained model on Hugging Face; we are NOT training our own
# image model, we're reusing an existing one.

# IMPORTANT LIMITATION (please read):
# Food-101 only covers 101 specific dish categories (mostly Western/
# restaurant-style dishes -- pizza, sushi, tacos, steak, etc). It does NOT
# have categories for plain ingredients like "raw broccoli" or "spinach".
# So image recognition will only successfully match a SUBSET of the foods
# in our nutrition dataset -- mainly prepared dishes, not raw produce.
# When there's no good match, we tell the user clearly and let them fall
# back to the manual dropdown instead of guessing wrong.
# """

# from transformers import pipeline
# from PIL import Image
# import pandas as pd

# # Model name on Hugging Face. Downloaded automatically on first use and
# # cached locally afterwards, so subsequent runs are fast.
# MODEL_NAME = "nateraw/food"

# # Lazily-loaded singleton so we don't reload the (somewhat large) model
# # from disk every single time a prediction is requested.
# _classifier = None


# def _get_classifier():
#     global _classifier
#     if _classifier is None:
#         _classifier = pipeline("image-classification", model=MODEL_NAME)
#     return _classifier


# def predict_food_from_image(image_path_or_file, top_k=3):
#     """
#     Run the pretrained Food-101 classifier on an image and return the
#     top-k most likely food labels with confidence scores.

#     Parameters
#     ----------
#     image_path_or_file : str or file-like object
#         Path to an image file, or a file-like object (e.g. what
#         Streamlit's st.file_uploader gives you).
#     top_k : int
#         How many top predictions to return.

#     Returns
#     -------
#     list of dicts: [{"label": "pizza", "score": 0.87}, ...]
#         Labels use underscores (Food-101 convention), e.g. "club_sandwich".
#     """
#     classifier = _get_classifier()

#     image = Image.open(image_path_or_file).convert("RGB")
#     results = classifier(image)

#     # Hugging Face pipeline already sorts by score descending.
#     top_results = results[:top_k]

#     return [
#         {"label": r["label"], "score": round(r["score"], 4)}
#         for r in top_results
#     ]


# def food101_label_to_search_term(label):
#     """
#     Food-101 labels look like 'club_sandwich' or 'beef_tartare'
#     (underscored, lowercase). Our nutrition dataset uses plain spaced
#     food names (e.g. 'club sandwich'). This converts the model's label
#     into a search-friendly term to look up against our dataset.
#     """
#     return label.replace("_", " ").strip().lower()


# def match_food_to_dataset(predicted_label, df, food_column="Food"):
#     """
#     Try to match a Food-101 predicted label against our nutrition
#     dataset. Tries an exact match first, then falls back to a partial
#     (substring) match, since naming conventions between Food-101 and our
#     dataset won't always line up perfectly.

#     Returns
#     -------
#     (matched_row_or_None, list_of_suggestion_strings)
#         If a confident match is found, matched_row is a one-row
#         DataFrame and suggestions is empty.
#         If no confident match is found, matched_row is None and
#         suggestions contains nearby candidate food names (could be
#         empty if nothing even loosely matches).
#     """
#     search_term = food101_label_to_search_term(predicted_label)
#     candidates_to_try = [search_term]

#     # Handle simple singular/plural mismatches (e.g. "spring rolls" in
#     # Food-101 vs "Spring roll" singular in our dataset), since this is
#     # a common source of otherwise-good matches being missed.
#     if search_term.endswith("s") and not search_term.endswith("ss"):
#         candidates_to_try.append(search_term[:-1])
#     else:
#         candidates_to_try.append(search_term + "s")

#     for term in candidates_to_try:
#         # 1. Exact match -- always the most reliable, try this first.
#         exact = df[df[food_column].str.lower() == term]
#         if not exact.empty:
#             return exact.iloc[[0]], []

#     for term in candidates_to_try:
#         # 2. Substring match. IMPORTANT: a long, unrelated dataset entry
#         # can "accidentally" contain our search term as a substring
#         # (e.g. searching "donuts" matching "caramel iced coffee dunkin
#         # donuts" -- a coffee drink, not a donut). To avoid this, among
#         # all substring matches we prefer the SHORTEST matching food
#         # name, since a shorter name is far more likely to be the plain,
#         # direct match (e.g. "donut") rather than an unrelated dish that
#         # happens to mention the word.
#         contains_match = df[
#             df[food_column].str.lower().str.contains(term, na=False, regex=False)
#         ]
#         if not contains_match.empty:
#             shortest_match = contains_match.loc[
#                 contains_match[food_column].str.len().idxmin()
#             ]
#             return pd.DataFrame([shortest_match]), []

#     # 3. Last resort: try matching on individual significant words, and
#     # return the candidates as SUGGESTIONS rather than an automatic
#     # match, since word-level matching is too loose to trust blindly.
#     words = [w for w in search_term.split() if len(w) > 3]
#     for word in words:
#         word_match = df[df[food_column].str.lower().str.contains(word, na=False, regex=False)]
#         if not word_match.empty:
#             suggestions = word_match[food_column].head(5).tolist()
#             return None, suggestions

#     return None, []









# ========================================================================================


"""
Image-based food recognition for the Diabetes Food Recommendation System.

WHAT THIS DOES:
Takes a photo of food uploaded by the user and identifies which food it
is, using a pretrained image classification model (no training required
from us). The predicted food name is then matched against our nutrition
dataset (diabetes_ready_dataset.csv) so the existing Random Forest model
and personalization logic can run on it exactly like a manually-selected
food would.

MODEL USED:
"nateraw/food" -- a Vision Transformer (ViT) fine-tuned on the Food-101
dataset (101 food categories, ~89% accuracy). This is a publicly
available pretrained model on Hugging Face; we are NOT training our own
image model, we're reusing an existing one.

IMPORTANT LIMITATION (please read):
Food-101 only covers 101 specific dish categories (mostly Western/
restaurant-style dishes -- pizza, sushi, tacos, steak, etc). It does NOT
have categories for plain ingredients like "raw broccoli" or "spinach".
So image recognition will only successfully match a SUBSET of the foods
in our nutrition dataset -- mainly prepared dishes, not raw produce.
When there's no good match, we tell the user clearly and let them fall
back to the manual dropdown instead of guessing wrong.
"""

from transformers import pipeline
from PIL import Image
import pandas as pd
import streamlit as st

# Model name on Hugging Face. Downloaded automatically on first use and
# cached locally afterwards, so subsequent runs are fast.
MODEL_NAME = "nateraw/food"


@st.cache_resource(show_spinner=False)
def _get_classifier():
    """
    Loads the pretrained image classifier ONCE per app session and
    reuses it on every later call.

    WHY THIS MATTERS: Streamlit re-runs the entire script top-to-bottom
    every time the user interacts with a widget (e.g. clicking "Identify
    Food"). Without @st.cache_resource, a plain module-level variable
    looked like it should only load once, but in practice the model
    could still end up being reloaded on each rerun, making every single
    click feel slow (re-loading a multi-hundred-MB model takes many
    seconds). @st.cache_resource is the Streamlit-recommended way to
    persist expensive objects like ML models across reruns within the
    same session.
    """
    return pipeline("image-classification", model=MODEL_NAME)


def predict_food_from_image(image_path_or_file, top_k=3):
    """
    Run the pretrained Food-101 classifier on an image and return the
    top-k most likely food labels with confidence scores.

    Parameters
    ----------
    image_path_or_file : str or file-like object
        Path to an image file, or a file-like object (e.g. what
        Streamlit's st.file_uploader gives you).
    top_k : int
        How many top predictions to return.

    Returns
    -------
    list of dicts: [{"label": "pizza", "score": 0.87}, ...]
        Labels use underscores (Food-101 convention), e.g. "club_sandwich".
    """
    classifier = _get_classifier()

    image = Image.open(image_path_or_file).convert("RGB")
    results = classifier(image)

    # Hugging Face pipeline already sorts by score descending.
    top_results = results[:top_k]

    return [
        {"label": r["label"], "score": round(r["score"], 4)}
        for r in top_results
    ]


def food101_label_to_search_term(label):
    """
    Food-101 labels look like 'club_sandwich' or 'beef_tartare'
    (underscored, lowercase). Our nutrition dataset uses plain spaced
    food names (e.g. 'club sandwich'). This converts the model's label
    into a search-friendly term to look up against our dataset.
    """
    return label.replace("_", " ").strip().lower()


def match_food_to_dataset(predicted_label, df, food_column="Food"):
    """
    Try to match a Food-101 predicted label against our nutrition
    dataset. Tries an exact match first, then falls back to a partial
    (substring) match, since naming conventions between Food-101 and our
    dataset won't always line up perfectly.

    Returns
    -------
    (matched_row_or_None, list_of_suggestion_strings)
        If a confident match is found, matched_row is a one-row
        DataFrame and suggestions is empty.
        If no confident match is found, matched_row is None and
        suggestions contains nearby candidate food names (could be
        empty if nothing even loosely matches).
    """
    search_term = food101_label_to_search_term(predicted_label)
    candidates_to_try = [search_term]

    # Handle simple singular/plural mismatches (e.g. "spring rolls" in
    # Food-101 vs "Spring roll" singular in our dataset), since this is
    # a common source of otherwise-good matches being missed.
    if search_term.endswith("s") and not search_term.endswith("ss"):
        candidates_to_try.append(search_term[:-1])
    else:
        candidates_to_try.append(search_term + "s")

    for term in candidates_to_try:
        # 1. Exact match -- always the most reliable, try this first.
        exact = df[df[food_column].str.lower() == term]
        if not exact.empty:
            return exact.iloc[[0]], []

    for term in candidates_to_try:
        # 2. Substring match. IMPORTANT: a long, unrelated dataset entry
        # can "accidentally" contain our search term as a substring
        # (e.g. searching "donuts" matching "caramel iced coffee dunkin
        # donuts" -- a coffee drink, not a donut). To avoid this, among
        # all substring matches we prefer the SHORTEST matching food
        # name, since a shorter name is far more likely to be the plain,
        # direct match (e.g. "donut") rather than an unrelated dish that
        # happens to mention the word.
        contains_match = df[
            df[food_column].str.lower().str.contains(term, na=False, regex=False)
        ]
        if not contains_match.empty:
            shortest_match = contains_match.loc[
                contains_match[food_column].str.len().idxmin()
            ]
            return pd.DataFrame([shortest_match]), []

    # 3. Last resort: try matching on individual significant words, and
    # return the candidates as SUGGESTIONS rather than an automatic
    # match, since word-level matching is too loose to trust blindly.
    words = [w for w in search_term.split() if len(w) > 3]
    for word in words:
        word_match = df[df[food_column].str.lower().str.contains(word, na=False, regex=False)]
        if not word_match.empty:
            suggestions = word_match[food_column].head(5).tolist()
            return None, suggestions

    return None, []