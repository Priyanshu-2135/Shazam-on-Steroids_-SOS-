
import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

DATASET_PATH = "features_3_sec.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "features_3_sec.csv not found. "
        "Place it in the same folder as music_genre.py."
    )

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# --------------------------------------------------
# 2. PREPARE DATA
# --------------------------------------------------

# Identify original song from segment filename
groups = df["filename"].astype(str).str.replace(
    r"\.\d+$", "", regex=True
)

# Remove labels, filenames, and non-feature columns
X = df.drop(
    columns=["label", "filename", "length"],
    errors="ignore"
)

# Keep only numerical features
X = X.select_dtypes(include=[np.number])

y = df["label"]

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

# Keep segments from the same original song together
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", X.shape[1])

# --------------------------------------------------
# 4. TRAIN RANDOM FOREST MODEL
# --------------------------------------------------

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# 5. EVALUATE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification report:")
print(classification_report(y_test, predictions))

# --------------------------------------------------
# 6. CONFUSION MATRIX
# --------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Music Genre Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 important features:")
print(importance.head(10))

importance.head(10).sort_values().plot(
    kind="barh",
    figsize=(9, 5)
)

plt.title("Top 10 Important Audio Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# --------------------------------------------------
# 8. SAVE TRAINED MODEL
# --------------------------------------------------

joblib.dump(
    {"model": model, "feature_columns": list(X.columns)},
    "music_genre_model.pkl"
)

print("\nModel saved as music_genre_model.pkl")
print("Graphs saved as PNG files.")
