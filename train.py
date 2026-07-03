# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # import pickle

# # # Load dataset
# # df = pd.read_csv("dataset/diabetes_food.csv")

# # # Features & Target
# # X = df[['Sugar', 'Carbs', 'Calories']]
# # y = df['Safe']

# # # Split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # # Model
# # model = RandomForestClassifier()
# # model.fit(X_train, y_train)

# # # Accuracy
# # print("Accuracy:", model.score(X_test, y_test))

# # # Save model
# # with open("model.pkl", "wb") as f:
# #     pickle.dump(model, f)

# # print("Model saved!")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report
# import pickle

# # Load dataset
# df = pd.read_csv("dataset/diabetes_ready_dataset.csv")

# # Features
# X = df[['Sugar', 'Carbs', 'Calories', 'Fiber', 'Protein', 'Fat', 'Sodium']]

# # Labels
# y = df['Label']

# # Convert text labels to numbers
# encoder = LabelEncoder()
# y = encoder.fit_transform(y)

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Model
# model = RandomForestClassifier()

# # Train
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Report
# print(classification_report(y_test, y_pred))

# # Save model
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# # Save encoder
# with open("label_encoder.pkl", "wb") as f:
#     pickle.dump(encoder, f)

# print("✅ Model trained successfully!")













import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("dataset/diabetes_ready_dataset.csv")

# Features
FEATURE_COLUMNS = ['Sugar', 'Carbs', 'Calories', 'Fiber', 'Protein', 'Fat', 'Sodium']
X = df[FEATURE_COLUMNS]

# Labels
y_raw = df['Label']

# Convert text labels to numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
# FIX 1: stratify=y ensures the train/test split preserves the same
# proportion of each label (Avoid/Moderate/Recommended/Highly Recommended)
# in both sets. Without this, a split could randomly under-represent a
# rarer label in the test set, making the evaluation metrics unreliable.
#
# FIX 2: random_state is fixed so re-running this script gives a
# reproducible split -- useful when comparing model versions.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# MODEL + HYPERPARAMETER TUNING
# -----------------------------
# FIX 3: instead of training a single RandomForestClassifier() with all
# default settings, we search over a small grid of reasonable
# hyperparameters and keep the best one (picked via cross-validation on
# the training set only, so the test set stays unseen until final
# evaluation).
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

base_model = RandomForestClassifier(random_state=42, class_weight="balanced")

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring="f1_weighted",  # weighted F1 is more informative than plain
                            # accuracy when classes are imbalanced
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

print("Best hyperparameters found:", grid_search.best_params_)

# -----------------------------
# EVALUATION
# -----------------------------

y_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
report_text = classification_report(
    y_test, y_pred, target_names=encoder.classes_
)
report_dict = classification_report(
    y_test, y_pred, target_names=encoder.classes_, output_dict=True
)

print(f"\nTest Accuracy: {test_accuracy:.4f}\n")
print(report_text)

# -----------------------------
# FIX 4: SAVE METRICS TO DISK
# -----------------------------
# Previously the classification report was only printed to the console
# and lost after the script finished. Now we save it to a JSON file so
# model performance can be tracked/compared across versions over time.
metrics_output = {
    "test_accuracy": test_accuracy,
    "best_hyperparameters": grid_search.best_params_,
    "classification_report": report_dict,
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics_output, f, indent=2)

# -----------------------------
# SAVE MODEL + ENCODER
# -----------------------------

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nModel trained and saved successfully!")
print("Metrics saved to model_metrics.json")