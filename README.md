# Netflix Audience Rating Classification

A Machine Learning project that predicts audience rating categories for Netflix titles using content-related features.

## Project Overview

This project explores Netflix content data and trains classification models to predict audience ratings. It includes dataset preparation, model comparison, hyperparameter tuning, model evaluation, and a Streamlit web application.

## Objectives

* Analyze Netflix audience rating categories.
* Prepare features for machine learning.
* Train and compare Decision Tree and Random Forest classifiers.
* Tune model hyperparameters.
* Evaluate predictions using classification metrics.
* Provide an interactive rating prediction interface.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

## Dataset

The dataset contains 8,790 Netflix titles and 10 columns.

Selected model features:

* Content type
* Country
* Release year
* Duration
* Genres

**Target:** `rating`

## Model Performance

| Metric         | Tuned Random Forest |
| -------------- | ------------------: |
| Test records   |               1,758 |
| Accuracy       |              51.76% |
| Macro F1-score |              0.3250 |

The dataset contains 14 rating categories, including some very rare classes. Performance varies across categories, and the reported results should be interpreted with this class imbalance in mind.

## Project Files

* `Dataset.csv` — Original dataset
* `check_dataset.py` — Dataset inspection
* `prepare_dataset.py` — Data preparation and splitting
* `train_models.py` — Initial model comparison
* `tune_model.py` — Hyperparameter tuning
* `evaluate_model.py` — Final model evaluation
* `predict_rating.py` — Sample prediction
* `app.py` — Streamlit web application
* `best_rating_model.pkl` — Tuned model
* `model_comparison.csv` — Initial model comparison results
* `hyperparameter_results.csv` — Tuning results
* `final_confusion_matrix.png` — Final evaluation visualization

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the web application:

```bash
python -m streamlit run app.py
```

