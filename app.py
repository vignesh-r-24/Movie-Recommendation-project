import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Advanced Movie Recommendation System using SVD")

# Load data
@st.cache_data
def load_data():
    ratings_df = pd.read_csv('ratings.dat',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python', delimiter='::', header=None, encoding='latin1')
    
    movies_df = pd.read_csv('movies.dat',
        names=['movie_id', 'title', 'genre'],
        engine='python', delimiter='::', header=None, encoding='latin1')
    
    return ratings_df, movies_df

# Build SVD model
@st.cache_data
def build_svd_model(ratings_df, movies_df):
    # Create user-movie rating matrix
    user_movie_matrix = ratings_df.pivot_table(
        index='user_id', 
        columns='movie_id', 
        values='rating',
        fill_value=0
    )
    
    # Apply SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd_matrix = svd.fit_transform(user_movie_matrix)
    
    return user_movie_matrix, svd_matrix

ratings_df, movies_df = load_data()
user_movie_matrix, svd_matrix = build_svd_model(ratings_df, movies_df)

# Sidebar
st.sidebar.header(" Settings")
recommendation_method = st.sidebar.radio("Select Method", 
    ["Genre-based", "Rating-based"])
    #  "User-based (SVD)"
    #  "Rating-based"])

st.sidebar.write("### Dataset Stats")
st.sidebar.metric("Total Users", ratings_df['user_id'].nunique())
st.sidebar.metric("Total Movies", len(movies_df))
st.sidebar.metric("Total Ratings", len(ratings_df))

col1, col2, col3, col4 = st.columns(4)

with col1:
    user_id = st.number_input("Enter User ID", min_value=1, value=1)

with col2:
    genre = st.text_input("Filter by Genre", value="Action")

with col3:
    n_recommend = st.number_input("Recommendations Count", min_value=1, max_value=20, value=5)

with col4:
    min_rating = st.number_input("Minimum Rating", 0.0, 5.0, 3.0)

if st.button(" Get Recommendations", use_container_width=True):
    st.divider()
    
    # Filter by genre
    filtered_movies = movies_df[movies_df['genre'].str.contains(genre, case=False, na=False)]
    
    if len(filtered_movies) == 0:
        st.error(f"No movies found for genre: {genre}")
    else:
        if recommendation_method == "Genre-based":
            # Get top-rated movies in genre
            genre_movies = filtered_movies['movie_id'].tolist()
            genre_ratings = ratings_df[ratings_df['movie_id'].isin(genre_movies)]
            avg_ratings = genre_ratings.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
            top_movies = avg_ratings[avg_ratings >= min_rating].head(n_recommend).index.tolist()
            recommended = movies_df[movies_df['movie_id'].isin(top_movies)]
        
        elif recommendation_method == "User-based (SVD)":
            # SVD-based recommendations
            if user_id <= len(svd_matrix):
                user_vec = svd_matrix[user_id - 1].reshape(1, -1)
                similarities = cosine_similarity(user_vec, svd_matrix)[0]
                similar_users = np.argsort(similarities)[-11:-1]
                
                similar_user_ratings = ratings_df[ratings_df['user_id'].isin(similar_users + 1)]
                recommendations = similar_user_ratings.groupby('movie_id')['rating'].mean()
                recommendations = recommendations[recommendations >= min_rating].sort_values(ascending=False)
                top_movies = recommendations.head(n_recommend).index.tolist()
                recommended = movies_df[movies_df['movie_id'].isin(top_movies)]
            else:
                st.warning(f"User ID {user_id} not found in dataset")
                recommended = filtered_movies.sample(n=min(n_recommend, len(filtered_movies)))
        
        else:  # Rating-based
            # Get highest-rated movies in genre
            genre_movies = filtered_movies['movie_id'].tolist()
            genre_ratings = ratings_df[ratings_df['movie_id'].isin(genre_movies)]
            avg_ratings = genre_ratings.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
            top_movies = avg_ratings[avg_ratings >= min_rating].head(n_recommend).index.tolist()
            recommended = movies_df[movies_df['movie_id'].isin(top_movies)]
        
        if len(recommended) == 0:
            st.warning(f"No recommendations found with minimum rating {min_rating}")
        else:
            st.success(f" Top {len(recommended)} Recommendations for User {user_id} ({genre}):")
            
            # Display as table with ratings
            display_df = recommended.copy()
            for idx, movie_id in enumerate(display_df['movie_id']):
                avg_rating = ratings_df[ratings_df['movie_id'] == movie_id]['rating'].mean()
                display_df.loc[display_df['movie_id'] == movie_id, 'avg_rating'] = f"{avg_rating:.2f}⭐"
            
            st.dataframe(display_df[['movie_id', 'title', 'genre', 'avg_rating']], use_container_width=True)

st.divider()
st.write("### Dataset Sample")
st.dataframe(movies_df.head(10), use_container_width=True)