"""
Netflix Content Recommendation System - Web Interface
Created with Streamlit

This is my frontend for the Netflix clustering project.
Allows anyone to search for titles and get recommendations!

Author: [Your Name]
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# For the ML part
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

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
    /* Main background */
    .main {
        background-color: #141414;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #E50914 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Text */
    p, div, label {
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #f40612;
        transform: scale(1.05);
    }
    
    /* Input boxes */
    .stTextInput>div>div>input {
        background-color: #333333;
        color: white;
        border: 1px solid #E50914;
        border-radius: 4px;
    }
    
    /* Selectbox */
    .stSelectbox>div>div>select {
        background-color: #333333;
        color: white;
    }
    
    /* Cards for recommendations */
    .recommendation-card {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #E50914;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    
    .recommendation-card:hover {
        transform: translateX(10px);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #E50914 !important;
        font-size: 2rem !important;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background-color: #1a472a !important;
        color: white !important;
    }
    
    .stInfo {
        background-color: #1a3a52 !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        color: white !important;
        border-radius: 4px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
        background-color: #333333;
        border-radius: 4px 4px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #E50914;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

@st.cache_data
def load_data():
    """
    Load the clustered dataset and prepare for recommendations
    This only runs once and caches the result for speed!
    """
    try:
        # Try to load the clustered results
        df = pd.read_csv('netflix_clustered_results.csv')
        st.sidebar.success("✓ Data loaded from saved results")
        return df
    except FileNotFoundError:
        # If no saved results, load original and process
        st.sidebar.warning("Loading original dataset...")
        df = pd.read_csv('NetflixSimple.csv')
        return df

@st.cache_resource
def prepare_similarity_matrix(df):
    """
    Prepare the similarity matrix for recommendations
    This is the heavy computation - only runs once!
    """
    with st.spinner("🔄 Building recommendation engine... (this takes ~30 seconds)"):
        
        # Fill missing values
        text_fields = ['director', 'cast', 'country', 'description', 'listed_in']
        for field in text_fields:
            df[field] = df[field].fillna('')
        
        # Create bag of content
        df['bag_of_content'] = (
            df['description'].astype(str) + ' ' +
            df['listed_in'].astype(str) + ' ' +
            df['director'].astype(str) + ' ' +
            df['cast'].astype(str) + ' ' +
            df['country'].astype(str)
        )
        
        # Preprocess text
        lemmatizer = WordNetLemmatizer()
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set(['the', 'is', 'at', 'which', 'on', 'a', 'an'])
        
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            processed_tokens = [
                lemmatizer.lemmatize(word)
                for word in tokens
                if word not in stop_words and len(word) > 2
            ]
            return ' '.join(processed_tokens)
        
        # Apply preprocessing (with progress bar)
        progress_bar = st.sidebar.progress(0)
        processed_texts = []
        for i, text in enumerate(df['bag_of_content']):
            processed_texts.append(preprocess_text(text))
            if i % 100 == 0:
                progress_bar.progress(i / len(df))
        
        df['processed_content'] = processed_texts
        progress_bar.progress(1.0)
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
        tfidf_matrix = vectorizer.fit_transform(df['processed_content'])
        
        # Calculate similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        st.sidebar.success("✓ Recommendation engine ready!")
        
        return df, similarity_matrix


# ============================================================================
# RECOMMENDATION FUNCTION
# ============================================================================

def get_recommendations(title, df, similarity_matrix, n=10):
    """
    Get recommendations for a given title
    """
    try:
        # Find the title (case-insensitive)
        idx = df[df['title'].str.lower() == title.lower()].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        
        # Get recommendations
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        recommendations = df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = scores
        
        return recommendations
    
    except IndexError:
        return None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    
    # Header with Netflix logo style
    st.markdown("""
        <h1 style='text-align: center; color: #E50914; font-size: 4rem; 
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            🎬 NETFLIX RECOMMENDER
        </h1>
        <p style='text-align: center; color: #ffffff; font-size: 1.2rem;'>
            Discover your next binge-worthy show using AI-powered content analysis
        </p>
        <hr style='border: 1px solid #E50914;'>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", 
                width=200)
        
        st.markdown("---")
        
        st.markdown("### 🎯 About This Project")
        st.info("""
        This recommendation system uses:
        - **NLP** to understand content
        - **K-Means Clustering** to group similar titles
        - **Cosine Similarity** for recommendations
        
        **No user ratings needed!** Pure content-based.
        """)
        
        st.markdown("---")
        
        # Load data
        df = load_data()
        df, similarity_matrix = prepare_similarity_matrix(df)
        
        st.markdown("### 📊 Dataset Stats")
        total_titles = len(df)
        n_movies = len(df[df['type'] == 'Movie']) if 'type' in df.columns else 0
        n_shows = len(df[df['type'] == 'TV Show']) if 'type' in df.columns else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Titles", f"{total_titles:,}")
            st.metric("Movies", f"{n_movies:,}")
        with col2:
            st.metric("TV Shows", f"{n_shows:,}")
            if 'cluster' in df.columns:
                st.metric("Clusters", df['cluster'].nunique())
        
        st.markdown("---")
        
        st.markdown("### 👨‍💻 Developer")
        st.markdown("""
        **[Your Name]**  
        Machine Learning Project  
        February 2026
        """)
    
    
    # Main content area - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Get Recommendations", 
        "📊 Explore Data", 
        "🎲 Random Pick",
        "ℹ️ How It Works"
    ])
    
    
    # ========================================================================
    # TAB 1: GET RECOMMENDATIONS
    # ========================================================================
    
    with tab1:
        st.markdown("## 🎬 Find Similar Content")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Search box with autocomplete
            search_query = st.text_input(
                "🔎 Search for a title:",
                placeholder="Type a movie or show name...",
                help="Start typing to see suggestions"
            )
            
            if search_query:
                # Filter titles that match
                matches = df[df['title'].str.contains(search_query, case=False, na=False)]
                
                if len(matches) > 0:
                    st.success(f"Found {len(matches)} matching title(s)")
                    
                    # Show matches
                    selected_title = st.selectbox(
                        "Select a title:",
                        matches['title'].tolist()
                    )
                    
                    # Show details of selected title
                    if selected_title:
                        selected_data = df[df['title'] == selected_title].iloc[0]
                        
                        st.markdown("---")
                        st.markdown(f"### 📺 {selected_title}")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(f"**Type:** {selected_data.get('type', 'N/A')}")
                        with col_b:
                            st.markdown(f"**Rating:** {selected_data.get('rating', 'N/A')}")
                        with col_c:
                            st.markdown(f"**Year:** {selected_data.get('release_year', 'N/A')}")
                        
                        st.markdown(f"**Genres:** {selected_data.get('listed_in', 'N/A')}")
                        
                        with st.expander("📖 View Description"):
                            st.write(selected_data.get('description', 'No description available'))
                        
                        st.markdown("---")
                        
                        # Get recommendations button
                        if st.button("🎯 Get Recommendations", use_container_width=True):
                            with st.spinner("Finding similar content..."):
                                recommendations = get_recommendations(
                                    selected_title, df, similarity_matrix, n=10
                                )
                                
                                if recommendations is not None:
                                    st.success("✓ Found 10 similar titles!")
                                    
                                    # Display recommendations
                                    st.markdown("### 🍿 You might also like:")
                                    
                                    for idx, row in recommendations.iterrows():
                                        similarity_pct = row['similarity_score'] * 100
                                        
                                        # Create card
                                        st.markdown(f"""
                                        <div class='recommendation-card'>
                                            <h4 style='color: #E50914; margin: 0;'>
                                                {row['title']}
                                            </h4>
                                            <p style='margin: 0.5rem 0;'>
                                                <strong>Match:</strong> {similarity_pct:.1f}% | 
                                                <strong>Type:</strong> {row.get('type', 'N/A')} | 
                                                <strong>Rating:</strong> {row.get('rating', 'N/A')}
                                            </p>
                                            <p style='margin: 0.5rem 0;'>
                                                <strong>Genres:</strong> {row.get('listed_in', 'N/A')[:80]}...
                                            </p>
                                            <p style='margin: 0.5rem 0; font-size: 0.9rem; color: #cccccc;'>
                                                {row.get('description', 'No description')[:150]}...
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                else:
                                    st.error("Could not generate recommendations. Please try another title.")
                
                else:
                    st.warning("No matches found. Try a different search term.")
        
        with col2:
            st.markdown("### 💡 Try These")
            
            # Show random popular titles
            if len(df) > 0:
                popular_sample = df.sample(min(8, len(df)), random_state=42)
                
                for _, title_row in popular_sample.iterrows():
                    if st.button(title_row['title'], key=f"try_{title_row['title']}", 
                               use_container_width=True):
                        st.session_state['selected_title'] = title_row['title']
                        st.rerun()
    
    
    # ========================================================================
    # TAB 2: EXPLORE DATA
    # ========================================================================
    
    with tab2:
        st.markdown("## 📊 Dataset Insights")
        
        # Content type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Content Type Distribution")
            if 'type' in df.columns:
                type_counts = df['type'].value_counts()
                st.bar_chart(type_counts)
        
        with col2:
            st.markdown("### Top 10 Genres")
            if 'listed_in' in df.columns:
                all_genres = df['listed_in'].str.split(',', expand=True).stack()
                all_genres = all_genres.str.strip()
                top_genres = all_genres.value_counts().head(10)
                st.bar_chart(top_genres)
        
        st.markdown("---")
        
        # Cluster analysis
        if 'cluster' in df.columns:
            st.markdown("### 🎯 Cluster Analysis")
            
            cluster_choice = st.selectbox(
                "Select a cluster to explore:",
                sorted(df['cluster'].unique())
            )
            
            cluster_data = df[df['cluster'] == cluster_choice]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Titles in this cluster", len(cluster_data))
                
                # Top genres in cluster
                st.markdown("**Top Genres:**")
                cluster_genres = cluster_data['listed_in'].str.split(',', expand=True).stack()
                cluster_genres = cluster_genres.str.strip()
                top_cluster_genres = cluster_genres.value_counts().head(5)
                for genre, count in top_cluster_genres.items():
                    st.write(f"• {genre}: {count}")
            
            with col2:
                # Sample titles
                st.markdown("**Sample Titles:**")
                sample_titles = cluster_data['title'].head(10)
                for i, title in enumerate(sample_titles, 1):
                    st.write(f"{i}. {title}")
        
        st.markdown("---")
        
        # Top countries
        st.markdown("### 🌍 Top Content-Producing Countries")
        if 'country' in df.columns:
            all_countries = df['country'].str.split(',', expand=True).stack()
            all_countries = all_countries.str.strip()
            top_countries = all_countries.value_counts().head(15)
            st.bar_chart(top_countries)
    
    
    # ========================================================================
    # TAB 3: RANDOM PICK
    # ========================================================================
    
    with tab3:
        st.markdown("## 🎲 Feeling Lucky?")
        st.markdown("Can't decide what to watch? Let me pick for you!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🎲 Pick a Random Title for Me!", use_container_width=True):
                random_title = df.sample(1).iloc[0]
                
                st.markdown("---")
                st.markdown(f"### 🎬 {random_title['title']}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Type:** {random_title.get('type', 'N/A')}")
                    st.markdown(f"**Rating:** {random_title.get('rating', 'N/A')}")
                with col_b:
                    st.markdown(f"**Year:** {random_title.get('release_year', 'N/A')}")
                    if 'cluster' in df.columns:
                        st.markdown(f"**Cluster:** {random_title.get('cluster', 'N/A')}")
                
                st.markdown(f"**Genres:** {random_title.get('listed_in', 'N/A')}")
                
                st.markdown("**Description:**")
                st.write(random_title.get('description', 'No description available'))
                
                st.markdown("---")
                
                # Get recommendations for this random pick
                if st.button("Show Similar Titles", use_container_width=True):
                    recommendations = get_recommendations(
                        random_title['title'], df, similarity_matrix, n=5
                    )
                    
                    if recommendations is not None:
                        st.markdown("### Similar titles:")
                        for _, row in recommendations.iterrows():
                            st.markdown(f"• **{row['title']}** ({row['similarity_score']:.2f})")
    
    
    # ========================================================================
    # TAB 4: HOW IT WORKS
    # ========================================================================
    
    with tab4:
        st.markdown("## ℹ️ How This System Works")
        
        st.markdown("""
        ### 🧠 The Technology Behind the Recommendations
        
        This is a **content-based recommendation system** that understands what movies and shows 
        are actually about, without needing user ratings or watch history.
        
        ---
        
        #### 📚 Step 1: Natural Language Processing (NLP)
        
        I process the text data (description, genres, cast, director) using:
        - **Tokenization**: Breaking text into words
        - **Stopword Removal**: Removing common words like "the", "is", "a"
        - **Lemmatization**: Converting words to their root form (e.g., "running" → "run")
        
        ---
        
        #### 🔢 Step 2: TF-IDF Vectorization
        
        Text is converted into numbers using **TF-IDF** (Term Frequency - Inverse Document Frequency):
        - Common words get LOW scores
        - Rare, distinctive words get HIGH scores
        - Example: "bollywood" gets high score in Indian movies
        
        ---
        
        #### 🎯 Step 3: K-Means Clustering
        
        Similar content is grouped into clusters:
        - Uses machine learning to find natural groupings
        - Helps organize the 7,000+ titles
        - Each cluster represents a content theme
        
        ---
        
        #### 📐 Step 4: Cosine Similarity
        
        When you select a title, I:
        1. Compare it with ALL other titles using cosine similarity
        2. Find the titles with highest similarity scores
        3. Return the top 10 matches
        
        **Similarity Score**: 0 = completely different, 1 = identical
        
        ---
        
        ### 🎓 Key Advantages
        
        ✅ **No Cold Start Problem**: Works for new content immediately  
        ✅ **Privacy-Friendly**: Doesn't need user data  
        ✅ **Transparent**: You can see WHY titles are similar (genres, themes)  
        ✅ **Cultural Awareness**: Groups content by region and language naturally  
        
        ---
        
        ### 📊 Technical Details
        
        - **Dataset**: 7,788 Netflix titles
        - **Features**: 5,000 TF-IDF features
        - **Algorithm**: K-Means Clustering + Cosine Similarity
        - **Libraries**: scikit-learn, NLTK, pandas
        - **Framework**: Streamlit for the web interface
        
        ---
        
        ### 👨‍💻 Built By
        
        **[Your Name]**  
        Machine Learning Final Project  
        February 2026
        
        This project demonstrates:
        - Natural Language Processing
        - Unsupervised Machine Learning
        - Content-Based Filtering
        - Web Application Development
        """)
        
        # Show sample code
        with st.expander("📝 View Sample Code"):
            st.code("""
# How recommendations work (simplified):

# 1. Load the similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# 2. Find the title's index
idx = df[df['title'] == 'Stranger Things'].index[0]

# 3. Get similarity scores with all other titles
sim_scores = list(enumerate(similarity_matrix[idx]))

# 4. Sort by similarity and get top 10
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

# 5. Return the recommended titles
recommendations = df.iloc[[i[0] for i in sim_scores]]
            """, language='python')
    
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #888888;'>
            Built with ❤️ using Machine Learning | Netflix Content Clustering Project 2026
        </p>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()

