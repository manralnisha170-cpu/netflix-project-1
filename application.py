"""
Netflix Content Recommendation System - Web Interface
FIXED: Proper color contrast (dark background, white text)

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
# FIXED CSS - PROPER COLORS (Dark background, White text)
# ============================================================================

st.markdown("""
    <style>
    /* Force dark background everywhere */
    .main, .block-container, [data-testid="stAppViewContainer"] {
        background-color: #141414 !important;
    }
    
    /* Sidebar dark */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    
    /* ALL TEXT WHITE - CRITICAL FIX */
    .main p, .main div, .main span, .main label, 
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] div,
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: #ffffff !important;
    }
    
    /* Headers - Netflix Red */
    h1, h2, h3, h4, h5, h6 {
        color: #E50914 !important;
    }
    
    /* Input fields */
    input, textarea, .stTextInput input {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #E50914 !important;
    }
    
    /* Select boxes */
    select, .stSelectbox select {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
    }
    
    .stButton > button:hover {
        background-color: #f40612 !important;
        transform: scale(1.05);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #ffffff !important;
        background-color: #333333 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #E50914 !important;
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #E50914 !important;
    }
    
    /* Info/Success boxes */
    .stAlert {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Dataframe */
    .dataframe {
        color: #ffffff !important;
    }
    
    /* Recommendation cards */
    .rec-card {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #E50914;
        margin: 1rem 0;
        color: #ffffff !important;
    }
    
    .rec-card h4 {
        color: #E50914 !important;
        margin: 0 0 10px 0;
    }
    
    .rec-card p {
        color: #ffffff !important;
        margin: 5px 0;
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
        df = pd.read_csv('netflix_clustered_results.csv')
        return df
    except:
        try:
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
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 0;'>🎬 NETFLIX RECOMMENDER</h1>
            <p style='font-size: 1.3rem; color: #ffffff;'>Discover your next favorite show</p>
        </div>
        <hr style='border: 2px solid #E50914; margin: 2rem 0;'>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h3 style='color: #E50914;'>🎯 About This App</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #1a1a1a; padding: 1rem; border-radius: 8px; color: #ffffff;'>
        <p style='color: #ffffff;'>This recommendation system uses <strong>AI</strong> to find similar Netflix content.</p>
        <p style='color: #ffffff;'><strong>Technology:</strong></p>
        <ul style='color: #ffffff;'>
            <li>NLP Text Processing</li>
            <li>TF-IDF Vectorization</li>
            <li>Cosine Similarity</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)
        
        # Load data
        with st.spinner("Loading data..."):
            df = load_data()
        
        if df is not None:
            st.success("✓ Data loaded successfully!")
            
            st.markdown("<h3 style='color: #E50914;'>📊 Dataset Stats</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Titles", f"{len(df):,}")
                if 'type' in df.columns:
                    movies = len(df[df['type'] == 'Movie'])
                    st.metric("Movies", f"{movies:,}")
            with col2:
                if 'type' in df.columns:
                    shows = len(df[df['type'] == 'TV Show'])
                    st.metric("TV Shows", f"{shows:,}")
                if 'cluster' in df.columns:
                    st.metric("Clusters", df['cluster'].nunique())
            
            # Prepare engine
            with st.spinner("Building recommendation engine..."):
                df, similarity_matrix = prepare_recommendation_engine(df)
            
            if similarity_matrix is not None:
                st.success("✓ Recommendation engine ready!")
        else:
            st.error("❌ Failed to load data")
            st.stop()
        
        st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; color: #888;'>
            <p><strong>Developer:</strong><br>[Your Name]</p>
            <p>ML Project 2026</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Get Recommendations", "📊 Explore Data", "ℹ️ How It Works"])
    
    # TAB 1: RECOMMENDATIONS
    with tab1:
        st.markdown("<h2 style='color: #E50914;'>🎬 Find Similar Content</h2>", unsafe_allow_html=True)
        
        st.markdown("<p style='color: #ffffff; font-size: 1.1rem;'>Search for any Netflix title to get personalized recommendations based on content similarity.</p>", unsafe_allow_html=True)
        
        # Search box
        search = st.text_input("🔎 Search for a title:", placeholder="Type a movie or show name...", key="search_box")
        
        if search:
            matches = df[df['title'].str.contains(search, case=False, na=False)]
            
            if len(matches) > 0:
                st.success(f"✓ Found {len(matches)} matching title(s)")
                
                selected = st.selectbox("📺 Select a title:", matches['title'].tolist())
                
                if selected:
                    # Show details
                    data = df[df['title'] == selected].iloc[0]
                    
                    st.markdown(f"<h3 style='color: #E50914;'>📺 {selected}</h3>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"<p style='color: #ffffff;'><strong>Type:</strong> {data.get('type', 'N/A')}</p>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<p style='color: #ffffff;'><strong>Rating:</strong> {data.get('rating', 'N/A')}</p>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"<p style='color: #ffffff;'><strong>Year:</strong> {data.get('release_year', 'N/A')}</p>", unsafe_allow_html=True)
                    
                    st.markdown(f"<p style='color: #ffffff;'><strong>Genres:</strong> {data.get('listed_in', 'N/A')}</p>", unsafe_allow_html=True)
                    
                    with st.expander("📖 View Full Description"):
                        st.markdown(f"<p style='color: #ffffff;'>{data.get('description', 'No description available')}</p>", unsafe_allow_html=True)
                    
                    st.markdown("<hr style='border: 1px solid #333; margin: 2rem 0;'>", unsafe_allow_html=True)
                    
                    # Get recommendations button
                    if st.button("🎯 Get Recommendations", use_container_width=True):
                        with st.spinner("🔍 Finding similar titles..."):
                            recs = get_recommendations(selected, df, similarity_matrix, n=10)
                            
                            if recs is not None and len(recs) > 0:
                                st.success("✓ Found 10 similar titles!")
                                
                                st.markdown("<h3 style='color: #E50914; margin-top: 2rem;'>🍿 You Might Also Like:</h3>", unsafe_allow_html=True)
                                
                                for idx, row in recs.iterrows():
                                    sim_pct = row['similarity'] * 100
                                    
                                    st.markdown(f"""
                                    <div class='rec-card'>
                                        <h4>{row['title']}</h4>
                                        <p><strong>Match:</strong> {sim_pct:.0f}% | 
                                           <strong>Type:</strong> {row.get('type', 'N/A')} | 
                                           <strong>Rating:</strong> {row.get('rating', 'N/A')}</p>
                                        <p><strong>Genres:</strong> {row.get('listed_in', 'N/A')}</p>
                                        <p style='font-size: 0.95rem; color: #cccccc;'>{row.get('description', 'No description')[:180]}...</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error("❌ Could not generate recommendations. Please try another title.")
            else:
                st.warning("⚠️ No matches found. Try a different search term or check spelling.")
        else:
            st.info("💡 **Tip:** Start typing to search from 7,000+ Netflix titles!")
    
    # TAB 2: EXPLORE
    with tab2:
        st.markdown("<h2 style='color: #E50914;'>📊 Dataset Insights</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #ffffff;'>Content Type Distribution</h3>", unsafe_allow_html=True)
            if 'type' in df.columns:
                type_counts = df['type'].value_counts()
                st.bar_chart(type_counts)
        
        with col2:
            st.markdown("<h3 style='color: #ffffff;'>Top 10 Genres</h3>", unsafe_allow_html=True)
            if 'listed_in' in df.columns:
                genres = df['listed_in'].str.split(',', expand=True).stack()
                genres = genres.str.strip()
                top_genres = genres.value_counts().head(10)
                st.bar_chart(top_genres)
        
        st.markdown("<hr style='border: 1px solid #333; margin: 2rem 0;'>", unsafe_allow_html=True)
        
        if 'country' in df.columns:
            st.markdown("<h3 style='color: #ffffff;'>🌍 Top Content-Producing Countries</h3>", unsafe_allow_html=True)
            countries = df['country'].str.split(',', expand=True).stack()
            countries = countries.str.strip()
            top_countries = countries.value_counts().head(15)
            st.bar_chart(top_countries)
        
        st.markdown("<hr style='border: 1px solid #333; margin: 2rem 0;'>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #ffffff;'>📋 Sample Titles</h3>", unsafe_allow_html=True)
        display_cols = ['title', 'type', 'rating', 'release_year']
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols].head(20), use_container_width=True)
    
    # TAB 3: HOW IT WORKS
    with tab3:
        st.markdown("<h2 style='color: #E50914;'>ℹ️ How This System Works</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='color: #ffffff;'>
        
        <h3 style='color: #E50914;'>🧠 The Technology Behind Recommendations</h3>
        
        <p>This is a <strong>content-based recommendation system</strong> that understands what movies and shows 
        are actually about, without needing user ratings or watch history.</p>
        
        <hr style='border: 1px solid #333;'>
        
        <h4 style='color: #E50914;'>📚 Step 1: Data Processing</h4>
        <ul>
            <li>Combines description, genres, cast, and director information</li>
            <li>Cleans and normalizes text data</li>
            <li>Removes special characters and converts to lowercase</li>
        </ul>
        
        <h4 style='color: #E50914;'>🔢 Step 2: Feature Engineering (TF-IDF)</h4>
        <ul>
            <li><strong>TF-IDF</strong> = Term Frequency - Inverse Document Frequency</li>
            <li>Converts text into numerical features</li>
            <li>Common words get LOW scores (e.g., "movie", "show")</li>
            <li>Rare, distinctive words get HIGH scores (e.g., "bollywood", "noir")</li>
        </ul>
        
        <h4 style='color: #E50914;'>📐 Step 3: Similarity Calculation</h4>
        <ul>
            <li>Uses <strong>Cosine Similarity</strong> to compare titles</li>
            <li>Measures the angle between feature vectors</li>
            <li>Score of 1.0 = identical, 0.0 = completely different</li>
            <li>Finds the top 10 most similar titles</li>
        </ul>
        
        <hr style='border: 1px solid #333;'>
        
        <h3 style='color: #E50914;'>🎯 Key Advantages</h3>
        
        <div style='background-color: #1a1a1a; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;'>
            <p>✅ <strong>No Cold Start Problem:</strong> Works for new content immediately</p>
            <p>✅ <strong>Privacy-Friendly:</strong> Doesn't need user data or watch history</p>
            <p>✅ <strong>Transparent:</strong> You can see WHY titles are similar</p>
            <p>✅ <strong>Cultural Awareness:</strong> Groups content by region and language naturally</p>
        </div>
        
        <hr style='border: 1px solid #333;'>
        
        <h3 style='color: #E50914;'>📊 Technical Specifications</h3>
        
        <table style='color: #ffffff; width: 100%;'>
            <tr><td><strong>Dataset Size:</strong></td><td>7,788 Netflix titles</td></tr>
            <tr><td><strong>Features:</strong></td><td>3,000 TF-IDF features</td></tr>
            <tr><td><strong>Algorithm:</strong></td><td>Cosine Similarity</td></tr>
            <tr><td><strong>Libraries:</strong></td><td>scikit-learn, pandas, NumPy</td></tr>
            <tr><td><strong>Framework:</strong></td><td>Streamlit</td></tr>
        </table>
        
        <hr style='border: 1px solid #333;'>
        
        <h3 style='color: #E50914;'>👨‍💻 Project Information</h3>
        
        <p><strong>Developer:</strong> [Your Name]</p>
        <p><strong>Project:</strong> Netflix Content Clustering & Recommendation</p>
        <p><strong>Date:</strong> February 2026</p>
        <p><strong>Course:</strong> Machine Learning Final Project</p>
        
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<hr style='border: 1px solid #333; margin-top: 3rem;'>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; color: #888888; padding: 1rem 0;'>
            🎬 Netflix Content Recommender | Built with Machine Learning | 2026
        </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
