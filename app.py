import streamlit as st
from recommender import HybridRecommender

st.set_page_config(page_title="Netflix-Style Hybrid Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Hybrid Movie & TV Show Recommender")
st.caption(
    "Content-based filtering (genre / cast / director / description) "
    "blended with a popularity & recency prior — built from scratch."
)


@st.cache_resource
def load_engine():
    return HybridRecommender("netflix_titles.csv")


engine = load_engine()

query = st.text_input("Search for a title you like", placeholder="e.g. Breaking Bad")

if query:
    matches = engine.search_titles(query)
    if matches.empty:
        st.warning("No titles found. Try a different search.")
    else:
        chosen = st.selectbox("Select the exact title", matches["title"].tolist())

        alpha = st.slider(
            "Blend: pure content similarity (left) ↔ pure popularity (right)",
            0.0, 1.0, 0.7, 0.05,
        )

        if st.button("Get Recommendations"):
            results = engine.recommend(chosen, top_n=10, alpha=alpha)
            if results is None or results.empty:
                st.error("Couldn't generate recommendations for that title.")
            else:
                st.subheader(f"Because you liked '{chosen}'")
                for _, row in results.iterrows():
                    st.markdown(
                        f"**{row['title']}**  ({row['type']}, {int(row['release_year'])})  \n"
                        f"*{row['listed_in']}*  \n"
                        f"Match score: `{row['score']:.2f}`"
                    )
                    st.divider()
else:
    st.info("Start by searching for a movie or show above.")

st.sidebar.header("About")
st.sidebar.write(
    "This is a hybrid recommender system: it combines TF-IDF content "
    "similarity with a popularity/recency prior, the same general "
    "approach production recommenders use when blending multiple signals."
)
st.sidebar.write("Dataset: public Netflix titles catalog (8,800+ titles).")