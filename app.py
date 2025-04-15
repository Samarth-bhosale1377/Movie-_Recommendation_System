
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from collections import Counter

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['title', 'genres', 'overview', 'popularity', 'vote_average', 'vote_count']]
    df.dropna(subset=['overview'], inplace=True)
    df.drop_duplicates(subset='title', inplace=True)
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
    df['genres'] = df['genres'].apply(lambda x: ' '.join(x))
    return df

df = load_data()

@st.cache_resource
def create_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = create_similarity(df)

def recommend(title, n=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualizations", "Recommend Movies"])

if menu == "Home":
    st.markdown("""
    Welcome to the **Movie Recommendation System**!  
    This app recommends movies using content-based filtering based on movie overviews from the **TMDB 5000 Movie Dataset**.
    
    👉 Go to the sidebar to explore the dataset, visualize patterns, or get personalized movie recommendations.
    """)
    st.markdown("[📂 Download Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)")

elif menu == "Data Exploration":
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe())

elif menu == "Visualizations":
    st.subheader("🎨 Genre Frequency")
    genre_counts = Counter(" ".join(df['genres']).split())
    genre_df = pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"]).sort_values(by="Count", ascending=False)

    fig1, ax1 = plt.subplots()
    sns.barplot(x="Genre", y="Count", data=genre_df, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("⭐ Vote Average Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['vote_average'], bins=20, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("📊 Popularity vs Vote Count")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x="vote_count", y="popularity", data=df, ax=ax3)
    st.pyplot(fig3)

elif menu == "Recommend Movies":
    st.subheader("🎬 Movie Recommendation")
    movie_list = df['title'].sort_values().tolist()
    selected_movie = st.selectbox("Select a Movie", movie_list)

    if st.button("Recommend"):
        recommendations = recommend(selected_movie)
        if recommendations:
            st.success(f"Movies similar to '{selected_movie}':")
            for movie in recommendations:
                st.markdown(f"- {movie}")
        else:
            st.warning("No recommendations found.")
