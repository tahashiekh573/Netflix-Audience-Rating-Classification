
import streamlit as st
import pandas as pd
from joblib import load

MODEL_PATH = "best_rating_model.pkl"

st.set_page_config(
    page_title="Netflix Rating Predictor",
    page_icon="🎬",
    layout="wide",
)

@st.cache_resource
def load_model():
    return load(MODEL_PATH)


st.title("🎬 Netflix Audience Rating Predictor")
st.write(
    "Predict a possible audience rating using Netflix content details "
    "and a trained Machine Learning model."
)

st.info(
    "This tool provides model predictions for educational purposes. "
    "Predictions may be incorrect."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Run tune_model.py first."
    )
    st.stop()

st.subheader("Enter Content Details")

col1, col2 = st.columns(2)

with col1:
    content_type = st.selectbox(
        "Content Type",
        ["Movie", "TV Show"],
    )

    country = st.text_input(
        "Country",
        value="United States",
    )

    release_year = st.number_input(
        "Release Year",
        min_value=1925,
        max_value=2035,
        value=2020,
        step=1,
    )

with col2:
    duration = st.text_input(
        "Duration",
        value="90 min" if content_type == "Movie" else "2 Seasons",
    )

    listed_in = st.text_input(
        "Genres",
        value="Documentaries",
        help="Example: Dramas, International Movies",
    )

if st.button("Predict Audience Rating", type="primary"):
    if not country.strip() or not duration.strip() or not listed_in.strip():
        st.warning("Please complete all fields before predicting.")
    else:
        input_data = pd.DataFrame(
            [{
                "type": content_type,
                "country": country.strip(),
                "release_year": int(release_year),
                "duration": duration.strip(),
                "listed_in": listed_in.strip(),
            }]
        )

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        st.success(f"Predicted Audience Rating: **{prediction}**")
        st.caption(
            "This is a machine learning prediction, not an official "
            "Netflix rating or a guaranteed result."
        )