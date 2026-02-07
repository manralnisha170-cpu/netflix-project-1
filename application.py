"""
Netflix Content Recommendation System - Web Interface
Working Version - No Blank Screen Issues

Author: [Your Name]
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import re

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - NETFLIX THEME
# ============================================================================

st.markdown("""
    <style>
    .main {
        background-color: #141414;
    }
    
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    h1, h2, h3 {
        color: #E50914 !important;
    }
    
    p, div, label, span {
        color: #ffffff !important;
    }
    
    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #f40612;
    }
    
    .recommendation-card {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #E50914;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        # Try clustered results first
        df = pd.read_csv('netflix_clustered_results.csv')
        return df
    except:
        try:
            # Try original dataset
            df = pd.read_csv('NetflixSimple.csv')
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None


@st.cache_resource
def prepare_recommendation_engine(df):
    """Prepare similarity matrix for recommendations"""
    
    if df is None:
        return None, None
    
    # Fill missing values
    text_fields = ['director', 'cast', 'country', 'description', 'listed_in']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].fillna('')
    
    # Create combined text
    df['combined_text'] = (
        df['description'].astype(str) + ' ' +
        df['listed_in'].astype(str) + ' ' +
        df['director'].astype(str) + ' ' +
        df['cast'].astype(str)
    )
    
    # Simple text preprocessing
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    df['combined_text'] = df['combined_text'].apply(clean_text)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    
    # Calculate similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return df, similarity_matrix


def get_recommendations(title, df, similarity_matrix, n=10):
    """Get recommendations for a title"""
    try:
        idx = df[df['title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        
        indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        recs = df.iloc[indices].copy()
        recs['similarity'] = scores
        return recs
    except:
        return None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    
    # Title
    st.markdown("""
        <h1 style='text-align: center; font-size: 3rem;'>
            🎬 NETFLIX RECOMMENDER
        </h1>
        <p style='text-align: center; font-size: 1.2rem;'>
            Discover your next favorite show
        </p>
        <hr style='border: 1px solid #E50914;'>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 About")
        st.info("""
        This system uses AI to recommend 
        Netflix content based on similarity.
        
        **Technology:**
        - NLP Text Processing
        - TF-IDF Vectorization
        - Cosine Similarity
        """)
        
        st.markdown("---")
        
        # Load data with progress
        with st.spinner("Loading data..."):
            df = load_data()
        
        if df is not None:
            st.success("✓ Data loaded!")
            
            st.markdown("### 📊 Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Titles", len(df))
            with col2:
                if 'type' in df.columns:
                    movies = len(df[df['type'] == 'Movie'])
                    st.metric("Movies", movies)
            
            # Prepare engine
            with st.spinner("Building engine..."):
                df, similarity_matrix = prepare_recommendation_engine(df)
            
            if similarity_matrix is not None:
                st.success("✓ Engine ready!")
        else:
            st.error("Failed to load data")
            st.stop()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Recommendations", "📊 Explore", "ℹ️ About"])
    
    with tab1:
        st.markdown("## Find Similar Content")
        
        # Search
        search = st.text_input("Search for a title:", placeholder="Type a movie or show name...")
        
        if search:
            matches = df[df['title'].str.contains(search, case=False, na=False)]
            
            if len(matches) > 0:
                st.success(f"Found {len(matches)} matches")
                
                selected = st.selectbox("Select title:", matches['title'].tolist())
                
                if selected:
                    # Show selected title details
                    data = df[df['title'] == selected].iloc[0]
                    
                    st.markdown(f"### {selected}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Type:** {data.get('type', 'N/A')}")
                    with col2:
                        st.write(f"**Rating:** {data.get('rating', 'N/A')}")
                    with col3:
                        st.write(f"**Year:** {data.get('release_year', 'N/A')}")
                    
                    st.write(f"**Genres:** {data.get('listed_in', 'N/A')}")
                    
                    with st.expander("Description"):
                        st.write(data.get('description', 'N/A'))
                    
                    st.markdown("---")
                    
                    # Get recommendations
                    if st.button("Get Recommendations", use_container_width=True):
                        with st.spinner("Finding similar titles..."):
                            recs = get_recommendations(selected, df, similarity_matrix)
                            
                            if recs is not None:
                                st.success("Found recommendations!")
                                
                                st.markdown("### 🍿 Similar Titles:")
                                
                                for idx, row in recs.iterrows():
                                    sim_pct = row['similarity'] * 100
                                    
                                    st.markdown(f"""
                                    <div class='recommendation-card'>
                                        <h4 style='color: #E50914;'>{row['title']}</h4>
                                        <p><strong>Match:</strong> {sim_pct:.0f}% | 
                                           <strong>Type:</strong> {row.get('type', 'N/A')}</p>
                                        <p><strong>Genres:</strong> {row.get('listed_in', 'N/A')}</p>
                                        <p style='font-size: 0.9rem;'>{row.get('description', '')[:150]}...</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error("Could not generate recommendations")
            else:
                st.warning("No matches found")
    
    with tab2:
        st.markdown("## Dataset Overview")
        
        if 'type' in df.columns:
            st.markdown("### Content Type Distribution")
            type_counts = df['type'].value_counts()
            st.bar_chart(type_counts)
        
        if 'listed_in' in df.columns:
            st.markdown("### Top Genres")
            genres = df['listed_in'].str.split(',', expand=True).stack()
            top_genres = genres.value_counts().head(10)
            st.bar_chart(top_genres)
        
        st.markdown("### Sample Titles")
        st.dataframe(df[['title', 'type', 'rating', 'release_year']].head(20))
    
    with tab3:
        st.markdown("""
        ## How It Works
        
        ### Technology Stack
        
        1. **Data Processing**
           - Combines description, genres, cast, director
           - Cleans and normalizes text
        
        2. **Feature Engineering**
           - TF-IDF Vectorization
           - Converts text to numerical features
        
        3. **Similarity Calculation**
           - Cosine Similarity
           - Measures content similarity
        
        4. **Recommendations**
           - Returns top 10 most similar titles
           - Based on content, not user ratings
        
        ### Benefits
        
        ✅ No user data needed  
        ✅ Works for new content  
        ✅ Culturally aware  
        ✅ Transparent results  
        
        ### Built With
        
        - Python
        - Streamlit
        - scikit-learn
        - pandas
        
        **Developer:** [Your Name]  
        **Project:** Netflix Content Clustering  
        **Date:** February 2026
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #888;'>
            Netflix Content Recommender | ML Project 2026
        </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
