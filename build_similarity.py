import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from movie import normalize_title

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
csv_path = os.path.join(BASE_DIR, "telugu_movie_20k_dataset.csv")
df = pd.read_csv(csv_path)

print("Columns:", df.columns.tolist())

# Normalize titles
df["normalized_title"] = df["movie_name"].apply(normalize_title)

# Build tags safely
tags = df["movie_name"].fillna("")

if "genre" in df.columns:
    tags = tags + " " + df["genre"].fillna("")

if "overview" in df.columns:
    tags = tags + " " + df["overview"].fillna("")

df["tags"] = tags

# Vectorize
cv = CountVectorizer(max_features=3000, stop_words="english")
vectors = cv.fit_transform(df["tags"])

# Similarity matrix
similarity = cosine_similarity(vectors)

# Save pickle (CORRECT)
sim_path = os.path.join(BASE_DIR, "similarity.pkl")
with open(sim_path, "wb") as f:
    pickle.dump(similarity, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ similarity.pkl created successfully")
print("Shape:", similarity.shape)
print("File size (MB):", os.path.getsize(sim_path) / 1024 / 1024)
