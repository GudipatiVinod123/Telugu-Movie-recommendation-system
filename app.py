import os
import streamlit as st
import pandas as pd
import pickle
from movie import recommend_movies, normalize_title

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="CineRec 🎬",
    page_icon="🎥",
    layout="centered"
)

# ---------- BASE DIRECTORY ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "telugu_movie_20k_dataset.csv")
    df = pd.read_csv(csv_path)
    df["normalized_title"] = df["movie_name"].apply(normalize_title)
    return df


@st.cache_resource
def load_similarity():
    sim_path = os.path.join(BASE_DIR, "similarity.pkl")
    with open(sim_path, "rb") as f:
        return pickle.load(f)



# ---------- LOAD FILES ----------
df = load_data()
similar = load_similarity()

# ---------- UI ----------
st.title("🎬 CineRec")
st.subheader("Content-Based Movie Recommendation System")

st.write(
    "Enter a movie name in **any format** (upper/lower case, spaces or no spaces). "
    "CineRec will handle it automatically."
)

movie_input = st.text_input(
    "Enter movie name:",
    placeholder="Example: jersey(2008), JERSEY 2008, Jersey (2008)"
)

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

# ---------- ACTION ----------
if st.button("🎯 Recommend"):
    if movie_input.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations = recommend_movies(
            movie_input=movie_input,
            df=df,
            similarity=similar,
            top_n=top_n
        )

        if not recommendations:
            st.error("Movie not found. Please check spelling.")
        else:
            st.success("Recommended Movies:")
            for i, movie in enumerate(recommendations, start=1):
                st.write(f"{i}. 🎬 {movie}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Built with ❤️ using Python, Pandas, Cosine Similarity & Streamlit")
