
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

# Load the saved model
saved = joblib.load("music_genre_model.pkl")
model = saved["model"]
feature_columns = saved["feature_columns"]

# Load dataset
df = pd.read_csv("features_3_sec.csv")

# Identify original songs so the split matches training
groups = df["filename"].astype(str).str.replace(
    r"\.\d+$", "", regex=True
)

X = df.drop(
    columns=["label", "filename", "length"],
    errors="ignore"
).select_dtypes(include=[np.number])

X = X[feature_columns]
y = df["label"]

# Recreate the same test split used in music.py
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print("\nMUSIC GENRE PREDICTION DEMO")
print("-" * 35)
print("Available test samples:", len(X_test))

while True:
    choice = input(
        "\nEnter a test sample number (0 to "
        + str(len(X_test) - 1)
        + "), or q to quit: "
    ).strip()

    if choice.lower() == "q":
        break

    try:
        index = int(choice)

        if not 0 <= index < len(X_test):
            print("Please enter a valid sample number.")
            continue

        sample = X_test.iloc[[index]]
        prediction = model.predict(sample)[0]
        actual = y_test.iloc[index]

        print("\nPredicted genre:", prediction)
        print("Actual genre:   ", actual)

        if prediction == actual:
            print("Result: Correct prediction")
        else:
            print("Result: Incorrect prediction")

    except ValueError:
        print("Enter a number or q.")

print("\nDemo ended.")
