# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import pickle

# # Load dataset
# df = pd.read_csv("dataset/diabetes_food.csv")

# # Features & Target
# X = df[['Sugar', 'Carbs', 'Calories']]
# y = df['Safe']

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Accuracy
# print("Accuracy:", model.score(X_test, y_test))

# # Save model
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# print("Model saved!")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("dataset/diabetes_ready_dataset.csv")

# Features
X = df[['Sugar', 'Carbs', 'Calories', 'Fiber', 'Protein', 'Fat', 'Sodium']]

# Labels
y = df['Label']

# Convert text labels to numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Report
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Model trained successfully!")