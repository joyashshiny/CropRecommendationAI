import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Features and label
X = data.drop("label", axis=1)
y = data["label"]

# Split for better generalization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stronger model for all crops
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Accuracy:", accuracy)

# Save
pickle.dump(model, open("model/crop_model.pkl", "wb"))

print("✅ Model saved successfully")