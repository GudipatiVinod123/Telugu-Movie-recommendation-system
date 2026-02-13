import re


def normalize_title(title: str) -> str:
    if not isinstance(title, str):
        return ""

    title = title.lower()
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"[^a-z0-9 ]", "", title)
    title = re.sub(r"\s+", "", title)

    return title.strip()


def recommend_movies(movie_input, df, similarity, top_n=5):
    if not movie_input or not movie_input.strip():
        return []

    normalized_input = normalize_title(movie_input)

    # Find movie index
    matches = df[df["normalized_title"] == normalized_input]

    if matches.empty:
        return []

    idx = matches.index[0]

    if idx >= similarity.shape[0]:
        return []

    # Similarity scores
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    seen = set()

    for i, score in sim_scores:
        movie_name = df.iloc[i]["movie_name"]

        # Skip same movie & duplicates
        if normalize_title(movie_name) == normalized_input:
            continue

        if movie_name not in seen:
            seen.add(movie_name)
            recommendations.append(movie_name)

        if len(recommendations) == top_n:
            break

    return recommendations
