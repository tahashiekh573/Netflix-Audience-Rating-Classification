
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "Dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = [
    "type",
    "country",
    "release_year",
    "duration",
    "listed_in",
]

TARGET = "rating"


def main():
    df = pd.read_csv(DATASET_PATH)

    # Select relevant features and target
    data = df[FEATURES + [TARGET]].copy()

    # Remove rows with missing values in selected columns
    data = data.dropna()

    X = data[FEATURES]
    y = data[TARGET]

    # Split data while preserving rating-category proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 55)
    print("TASK 3 - FEATURE SELECTION")
    print("=" * 55)

    print(f"\nTotal usable records: {len(data)}")
    print(f"Selected features: {FEATURES}")
    print(f"Target column: {TARGET}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print("\nTraining rating distribution:")
    print(y_train.value_counts())

    print("\nTesting rating distribution:")
    print(y_test.value_counts())

    # Save prepared datasets
    train_data = X_train.copy()
    train_data[TARGET] = y_train

    test_data = X_test.copy()
    test_data[TARGET] = y_test

    train_data.to_csv("train_dataset.csv", index=False)
    test_data.to_csv("test_dataset.csv", index=False)

    print("\nSaved: train_dataset.csv")
    print("Saved: test_dataset.csv")


if __name__ == "__main__":
    main()