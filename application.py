"""
Netflix Content Clustering & Recommendation System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #564d4d;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎬 Netflix Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover similar content using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses **unsupervised machine learning** to recommend Netflix content based on:
    - Content descriptions
    - Genres
    - Cast and directors
    - Countries
    
    **Powered by:**
    - TF-IDF Vectorization
    - K-Means Clustering
    - Cosine Similarity
    """)
    
    st.header("How it works")
    st.markdown("""
    1. Select a movie/show from the dropdown
    2. Choose number of recommendations
    3. Get similar content instantly!
    """)

# Load data function with caching
@st.cache_data
def load_data():
    """Load the Netflix dataset"""
    try:
        df = pd.read_csv('netflix_titles.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found! Please upload 'netflix_titles.csv'")
        return None

# Load pre-computed models (if available)
@st.cache_resource
def load_models():
    """Load TF-IDF matrix and other models"""
    try:
        with open('tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        return tfidf_matrix
    except FileNotFoundError:
        return None

# Recommendation function
def get_recommendations(title, df, tfidf_matrix, n_recommendations=10):
    """Get top N similar titles based on cosine similarity"""
    if title not in df['title'].values:
        return None
    
    idx = df[df['title'] == title].index[0]
    title_vector = tfidf_matrix[idx]
    similarities = cosine_similarity(title_vector, tfidf_matrix)
    similarity_scores = similarities[0]
    similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
    
    recommendations = df.iloc[similar_indices][['title', 'type', 'listed_in', 'description', 'rating', 'release_year']].copy()
    recommendations['similarity_score'] = similarity_scores[similar_indices]
    
    return recommendations

# Main app
def main():
    df = load_data()
    
    if df is None:
        st.warning("Please upload the Netflix dataset to continue.")
        uploaded_file = st.file_uploader("Upload netflix_titles.csv", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset loaded successfully!")
        else:
            st.stop()
    
    tfidf_matrix = load_models()
    
    if tfidf_matrix is None:
        st.warning("⚠️ Pre-computed models not found. Computing on-the-fly...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        with st.spinner("Computing TF-IDF features..."):
            df['content_text'] = (
                df['title'].fillna('') + ' ' +
                df['director'].fillna('') + ' ' +
                df['cast'].fillna('') + ' ' +
                df['country'].fillna('') + ' ' +
                df['listed_in'].fillna('') + ' ' +
                df['description'].fillna('')
            )
            
            df['content_text'] = df['content_text'].str.lower().str.replace('[^\w\s]', '', regex=True)
            
            tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(df['content_text'])
            
            st.success("✅ TF-IDF computation complete!")
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Titles", len(df))
    with col2:
        st.metric("Movies", len(df[df['type'] == 'Movie']))
    with col3:
        st.metric("TV Shows", len(df[df['type'] == 'TV Show']))
    with col4:
        st.metric("Countries", df['country'].nunique())
    
    st.markdown("---")
    
    # User input
    st.subheader("🔍 Find Similar Content")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_title = st.selectbox(
            "Select a movie or TV show:",
            options=sorted(df['title'].unique()),
            index=0
        )
    
    with col2:
        n_recommendations = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
    
    # Get recommendations
    if st.button("🎯 Get Recommendations", type="primary"):
        with st.spinner("Finding similar content..."):
            recommendations = get_recommendations(
                selected_title, 
                df, 
                tfidf_matrix, 
                n_recommendations
            )
            
            if recommendations is None:
                st.error(f"Title '{selected_title}' not found in dataset.")
            else:
                st.success(f"✅ Found {len(recommendations)} recommendations for **{selected_title}**")
                
                # Show original title details
                original = df[df['title'] == selected_title].iloc[0]
                
                with st.expander("📺 Selected Title Details", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Type:** {original['type']}")
                        st.write(f"**Rating:** {original['rating']}")
                        st.write(f"**Release Year:** {original['release_year']}")
                    with col2:
                        st.write(f"**Genres:** {original['listed_in']}")
                        st.write(f"**Country:** {original['country']}")
                    
                    st.write(f"**Description:** {original['description']}")
                
                st.markdown("---")
                
                # Display recommendations
                st.subheader("🎬 Recommended Titles")
                
                for idx, row in recommendations.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"### {row['title']}")
                            st.write(f"**Description:** {row['description'][:200]}...")
                            st.write(f"**Genres:** {row['listed_in']}")
                        
                        with col2:
                            st.metric("Similarity", f"{row['similarity_score']:.2%}")
                            st.write(f"**Type:** {row['type']}")
                        
                        with col3:
                            st.write(f"**Rating:** {row['rating']}")
                            st.write(f"**Year:** {row['release_year']}")
                        
                        st.markdown("---")
                
                # Download recommendations
                csv = recommendations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations as CSV",
                    data=csv,
                    file_name=f"recommendations_{selected_title.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit | Powered by Machine Learning</p>
        <p>Netflix Content Clustering & Recommendation System</p>
    </div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
