
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

TARGET = "rating"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "type",
    "country",
    "duration",
    "listed_in",
]

NUMERICAL_FEATURES = ["release_year"]


def build_preprocessor():
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )


def main():
    train_data = pd.read_csv("train_dataset.csv")
    test_data = pd.read_csv("test_dataset.csv")

    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]

    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    best_model_name = None
    best_model = None
    best_f1 = -1

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1,
        })

        print(f"\n{'=' * 55}")
        print(name)
        print(f"{'=' * 55}")
        print(f"Accuracy:        {accuracy:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall:    {recall:.4f}")
        print(f"Macro F1-score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0,
        ))

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = pipeline
            best_predictions = predictions

        if name == "Random Forest":
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                xticks_rotation="vertical",
                cmap="Blues",
            )
            plt.title("Random Forest - Confusion Matrix")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=150)
            plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison.csv", index=False)

    print(f"\n{'=' * 55}")
    print("MODEL COMPARISON")
    print(f"{'=' * 55}")
    print(results_df.round(4).to_string(index=False))
    print(f"\nBest model by macro F1-score: {best_model_name}")

    print("\nSaved: model_comparison.csv")
    print("Saved: confusion_matrix.png")


if __name__ == "__main__":
    main()