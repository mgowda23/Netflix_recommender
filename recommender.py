"""
Hybrid Netflix-style recommendation engine.

Combines two signals into a single score for each candidate title:

1. CONTENT SIMILARITY  - cosine similarity between TF-IDF vectors built
   from each title's genres, cast, director and description. This captures
   "items that are *like* what you picked".

2. POPULARITY / RECENCY PRIOR - a normalized score that rewards titles
   which are newer and belong to genres that appear more often in the
   catalog (a cheap stand-in for "lots of other users watched/rated this
   highly", since the public dataset has no real per-user ratings).

The final ranking blends both with a tunable weight `alpha`, which is the
classic recipe real hybrid recommenders use: a content/similarity term
plus a collaborative/popularity term.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    def __init__(self, csv_path: str):
        self.df = self._load_and_clean(csv_path)
        self._build_content_model()
        self._build_popularity_prior()

    # ------------------------------------------------------------------ #
    # Data prep
    # ------------------------------------------------------------------ #
    def _load_and_clean(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        for col in ["director", "cast", "country", "listed_in", "description"]:
            df[col] = df[col].fillna("")
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0)
        df = df.reset_index(drop=True)

        # one combined "soup" of text features per title, used for TF-IDF
        df["soup"] = (
            df["listed_in"].str.replace(",", " ", regex=False) + " " +
            df["cast"].apply(lambda x: " ".join(x.split(",")[:4])) + " " +
            df["director"] + " " +
            df["description"]
        ).str.lower()

        return df

    def _build_content_model(self):
        tfidf = TfidfVectorizer(stop_words="english", min_df=2)
        self.tfidf_matrix = tfidf.fit_transform(self.df["soup"])
        self.vectorizer = tfidf

    def _build_popularity_prior(self):
        # recency, scaled 0-1
        year = self.df["release_year"]
        recency = (year - year.min()) / max(year.max() - year.min(), 1)

        # genre frequency, scaled 0-1 (proxy for "broadly popular categories")
        genre_counts = self.df["listed_in"].str.split(",").explode().str.strip().value_counts()
        def genre_score(genres_str):
            genres = [g.strip() for g in genres_str.split(",") if g.strip()]
            if not genres:
                return 0
            return sum(genre_counts.get(g, 0) for g in genres) / len(genres)
        raw_genre_score = self.df["listed_in"].apply(genre_score)
        genre_score_norm = (raw_genre_score - raw_genre_score.min()) / max(
            raw_genre_score.max() - raw_genre_score.min(), 1
        )

        self.df["popularity_prior"] = 0.5 * recency + 0.5 * genre_score_norm

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search_titles(self, query: str, limit: int = 10):
        mask = self.df["title"].str.contains(query, case=False, na=False)
        return self.df.loc[mask, ["title", "type", "release_year"]].head(limit)

    def recommend(self, title: str, top_n: int = 10, alpha: float = 0.7):
        """
        alpha: weight on content similarity (0-1).
               (1 - alpha) is the weight on the popularity prior.
        """
        matches = self.df.index[self.df["title"].str.lower() == title.lower()]
        if len(matches) == 0:
            return None
        idx = matches[0]

        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        pop = self.df["popularity_prior"].to_numpy()

        # normalize similarity to 0-1 for fair blending with popularity
        sims_norm = (sims - sims.min()) / max(sims.max() - sims.min(), 1e-9)

        final_score = alpha * sims_norm + (1 - alpha) * pop
        result = self.df.copy()
        result["score"] = final_score
        result = result[result.index != idx]
        result = result.sort_values("score", ascending=False).head(top_n)
        return result[["title", "type", "release_year", "listed_in", "score"]]
