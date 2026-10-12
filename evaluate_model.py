
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    classification_report,
)

TARGET = "rating"


def main():
    test_data = pd.read_csv("test_dataset.csv")

    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    model = load("best_rating_model.pkl")
    predictions = model.predict(X_test)

    print("=" * 55)
    print("FINAL MODEL EVALUATION")
    print("=" * 55)

    print(f"Test records: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Macro F1-score: "
        f"{f1_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation="vertical",
        cmap="Blues",
        values_format="d",
    )

    plt.title("Tuned Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("final_confusion_matrix.png", dpi=150)
    plt.close()

    print("\nSaved: final_confusion_matrix.png")


if __name__ == "__main__":
    main()