
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from joblib import dump

TARGET = "rating"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "type",
    "country",
    "duration",
    "listed_in",
]

NUMERICAL_FEATURES = ["release_year"]


def main():
    train_data = pd.read_csv("train_dataset.csv")
    test_data = pd.read_csv("test_dataset.csv")

    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]

    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 20, 30],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", 0.5],
        "classifier__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_grid,
        n_iter=6,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Starting Random Forest hyperparameter tuning...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    print("\nBest Parameters:")
    print(search.best_params_)

    print(f"\nBest Cross-Validation Macro F1: {search.best_score_:.4f}")

    print("\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Macro Precision: "
        f"{precision_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        "Macro Recall: "
        f"{recall_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        "Macro F1-score: "
        f"{f1_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    dump(best_model, "best_rating_model.pkl")
    pd.DataFrame(search.cv_results_).to_csv(
        "hyperparameter_results.csv",
        index=False,
    )

    print("\nSaved: best_rating_model.pkl")
    print("Saved: hyperparameter_results.csv")


if __name__ == "__main__":
    main()