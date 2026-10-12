
import pandas as pd
from joblib import load

MODEL_PATH = "best_rating_model.pkl"

FEATURES = [
    "type",
    "country",
    "release_year",
    "duration",
    "listed_in",
]


def predict_rating(
    content_type,
    country,
    release_year,
    duration,
    listed_in,
):
    model = load(MODEL_PATH)

    input_data = pd.DataFrame(
        [
            {
                "type": content_type,
                "country": country,
                "release_year": int(release_year),
                "duration": duration,
                "listed_in": listed_in,
            }
        ],
        columns=FEATURES,
    )

    prediction = model.predict(input_data)[0]

    print("\n" + "=" * 40)
    print("NETFLIX AUDIENCE RATING PREDICTION")
    print("=" * 40)
    print(f"Content Type : {content_type}")
    print(f"Country      : {country}")
    print(f"Release Year : {release_year}")
    print(f"Duration     : {duration}")
    print(f"Genres       : {listed_in}")
    print("-" * 40)
    print(f"Predicted Rating: {prediction}")
    print("=" * 40)


if __name__ == "__main__":
    predict_rating(
        content_type="Movie",
        country="United States",
        release_year=2020,
        duration="90 min",
        listed_in="Documentaries",
    )