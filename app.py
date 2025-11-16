import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ===============================
# 🧠 Load Dataset from Kaggle
# ===============================
@st.cache_data
def load_data():
    try:
        # Load the Kaggle dataset
        df = pd.read_csv("anime.csv")
        
        # Create the standardized dataframe with correct column names
        result_df = pd.DataFrame({
            'Anime': df['Name'],
            'Genre': df['Genres'],
            'Rating': df['Score'],
            'Description': df['Name'] + " - " + df['Genres']  # Simulated descriptions
        })
        
        # Clean and prepare the data
        result_df["Genre"] = result_df["Genre"].fillna("Unknown")
        result_df["Anime"] = result_df["Anime"].fillna("Unknown Anime")
        result_df["Description"] = result_df["Description"].fillna("No description available")
        
        # Convert Rating to numeric and handle missing values
        result_df["Rating"] = pd.to_numeric(result_df["Rating"], errors='coerce')
        result_df["Rating"] = result_df["Rating"].clip(1, 10)
        result_df["Rating"] = result_df["Rating"].fillna(result_df["Rating"].mean() if result_df["Rating"].notna().any() else 7.0)
        
        # Remove duplicates and filter out invalid entries
        result_df = result_df.drop_duplicates(subset=['Anime'])
        result_df = result_df[result_df['Anime'] != "Unknown Anime"]
        result_df = result_df[result_df['Rating'] > 0]
        
        # Prepare data for Naïve Bayes classification
        # Create a simplified genre classification (first genre as main category)
        result_df['Main_Genre'] = result_df['Genre'].apply(
            lambda x: x.split(',')[0].strip() if pd.notna(x) and x != "Unknown" else "Unknown"
        )
        
        print(f"Loaded {len(result_df)} anime entries")
        print(f"Rating range: {result_df['Rating'].min():.2f} - {result_df['Rating'].max():.2f}")
        print(f"Unique genres for classification: {result_df['Main_Genre'].nunique()}")
        
        return result_df
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Fallback to original small dataset
        try:
            df_fallback = pd.read_csv("anime_dataset.csv")
            df_fallback['Description'] = df_fallback['Anime'] + " - " + df_fallback['Genre']
            df_fallback['Main_Genre'] = df_fallback['Genre'].apply(
                lambda x: x.split(',')[0].strip() if pd.notna(x) else "Unknown"
            )
            st.info("Using fallback dataset")
            return df_fallback
        except:
            # Ultimate fallback - create minimal dataset
            st.warning("Using minimal fallback dataset")
            return pd.DataFrame({
                'Anime': ['Naruto', 'One Piece', 'Attack on Titan', 'Death Note', 'Your Name'],
                'Genre': ['Action, Adventure', 'Action, Adventure', 'Action, Fantasy', 'Thriller, Mystery', 'Romance, Drama'],
                'Rating': [8.2, 8.8, 9.0, 9.1, 9.2],
                'Description': [
                    'Ninja adventures and battles',
                    'Pirate adventures and friendships', 
                    'Titan battles and survival',
                    'Mystery thriller with supernatural elements',
                    'Romantic drama with supernatural twist'
                ],
                'Main_Genre': ['Action', 'Action', 'Action', 'Thriller', 'Romance']
            })

# ===============================
# 🧩 Naïve Bayes Genre Classifier
# ===============================
class AnimeGenreClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = MultinomialNB()
        self.is_trained = False
        self.genre_labels = None
        
    def prepare_data(self, df):
        """Prepare data for genre classification"""
        # Use descriptions as features and main genre as target
        X = df['Description'].fillna('')
        y = df['Main_Genre']
        
        # Filter out genres with too few samples
        genre_counts = y.value_counts()
        valid_genres = genre_counts[genre_counts >= 5].index
        mask = y.isin(valid_genres)
        
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        self.genre_labels = sorted(y_filtered.unique())
        return train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered)
    
    def train(self, X_train, y_train):
        """Train the Naïve Bayes classifier"""
        if len(X_train) == 0:
            raise ValueError("No training data available")
            
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)
        self.is_trained = True
        
    def predict(self, descriptions):
        """Predict genres for given descriptions"""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")
            
        descriptions_vec = self.vectorizer.transform(descriptions)
        predictions = self.classifier.predict(descriptions_vec)
        probabilities = self.classifier.predict_proba(descriptions_vec)
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate classifier performance"""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")
            
        predictions, probabilities = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        cm = confusion_matrix(y_test, predictions, labels=self.genre_labels)
        
        return accuracy, report, cm, predictions

# ===============================
# 📊 Initialize Components
# ===============================
df = load_data()

# Initialize classifier
genre_classifier = AnimeGenreClassifier()

# Train-test split for classification
try:
    X_train, X_test, y_train, y_test = genre_classifier.prepare_data(df)
    if len(X_train) > 0:
        genre_classifier.train(X_train, y_train)
        classification_ready = True
    else:
        classification_ready = False
        st.warning("Insufficient data for genre classification training")
except Exception as e:
    st.error(f"Error in classifier setup: {e}")
    classification_ready = False

# Recommendation system similarity matrix
try:
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["Genre"].fillna(""))
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
except Exception as e:
    st.error(f"Error in TF-IDF processing: {e}")
    similarity = np.eye(len(df))

# ===============================
# 🎨 Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="🎌 Anime ML Explorer", 
    layout="wide",
    page_icon="🎌"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    h1, h2, h3 {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        margin-bottom: 1rem !important;
    }
    
    .anime-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .anime-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(78, 205, 196, 0.3);
    }
    
    .small-text {
        font-size: 14px;
        color: #e0e0e0;
        margin: 5px 0;
    }
    
    .rating-star {
        color: #FFD93D;
        font-weight: bold;
        font-size: 16px;
    }
    
    .stats-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .made-by {
        font-size: 1.2em;
        font-weight: bold;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 🎌 Header Section
# ===============================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3.5em; margin-bottom: 0.5em;'>🎌 ANIME ML EXPLORER - CI 2</h1>
            <p style='font-size: 1.2em; color: #cccccc; max-width: 600px; margin: 0 auto;'>
                Advanced anime discovery with recommendation system and Naïve Bayes genre classification
            </p>
        </div>
    """, unsafe_allow_html=True)

# ===============================
# 🔍 Sidebar Filters
# ===============================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='margin-bottom: 0;'>🎯 FILTERS</h2>
            <div style='height: 3px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); width: 50%; margin: 0.5rem auto; border-radius: 2px;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    app_mode = st.selectbox("Select Mode", 
                           ["🎯 Recommendations", "🤖 Genre Classification", "📊 Model Evaluation"])
    
    # Common filters
    st.markdown("**🎭 SELECT GENRE**")
    all_genres = set()
    for genres in df["Genre"].fillna("").str.split(",", expand=True).stack().str.strip():
        if genres and genres != "Unknown":
            all_genres.add(genres)
    all_genres = sorted(all_genres)
    selected_genre = st.selectbox("", [""] + all_genres, label_visibility="collapsed")
    
    st.markdown("**⭐ MINIMUM RATING**")
    rating_min = max(1.0, df["Rating"].min())
    rating_max = min(10.0, df["Rating"].max())
    min_rating = st.slider("", float(rating_min), float(rating_max), 7.0, 0.1, label_visibility="collapsed")
    
    # Mode-specific filters
    if app_mode == "🎯 Recommendations":
        st.markdown("**🔍 SELECT ANIME FOR RECOMMENDATIONS**")
        anime_list = [""] + df["Anime"].sort_values().tolist()
        selected_anime = st.selectbox("", anime_list, label_visibility="collapsed")
        num_results = st.slider("**📊 NUMBER OF RECOMMENDATIONS**", 3, 12, 6)
    
    elif app_mode == "🤖 Genre Classification":
        st.markdown("**📝 ENTER ANIME DESCRIPTION**")
        custom_description = st.text_area("", 
                                        "Enter anime description for genre classification...",
                                        height=100,
                                        label_visibility="collapsed")
        
    # Stats Section
    st.markdown("---")
    st.markdown("**📈 DATABASE STATS**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class='stats-card'>
                <div style='font-size: 1.5em; font-weight: bold; color: #4ECDC4;'>📺</div>
                <div style='font-size: 0.9em;'>Total Anime</div>
                <div style='font-size: 1.2em; font-weight: bold;'>{len(df):,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='stats-card'>
                <div style='font-size: 1.5em; font-weight: bold; color: #FFD93D;'>⭐</div>
                <div style='font-size: 0.9em;'>Avg Rating</div>
                <div style='font-size: 1.2em; font-weight: bold;'>{df['Rating'].mean():.2f}</div>
            </div>
        """, unsafe_allow_html=True)

# ===============================
# 📋 Apply Common Filters
# ===============================
filtered = df.copy()
if selected_genre:
    filtered = filtered[filtered["Genre"].str.lower().str.contains(selected_genre.lower(), na=False)]
filtered = filtered[filtered["Rating"] >= min_rating]

# ===============================
# 🎯 RECOMMENDATION SYSTEM
# ===============================
if app_mode == "🎯 Recommendations":
    
    # Anime Collection Display
    st.markdown(f"""
        <div style='text-align: center; margin: 2rem 0;'>
            <h2>📺 ANIME COLLECTION</h2>
            <p style='color: #cccccc;'>Showing {len(filtered)} out of {len(df):,} anime</p>
            <div style='height: 3px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); width: 30%; margin: 0.5rem auto; border-radius: 2px;'></div>
        </div>
    """, unsafe_allow_html=True)

    if not filtered.empty:
        page_size = 9
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        
        if total_pages > 1:
            page = st.slider("**Navigate Pages**", 1, total_pages, 1, key="page_slider")
        else:
            page = 1
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered))
        current_page_data = filtered.iloc[start_idx:end_idx]
        
        cols = st.columns(3)
        for i, (_, anime) in enumerate(current_page_data.iterrows()):
            with cols[i % 3]:
                anime_name = str(anime['Anime'])
                genres = str(anime['Genre'])
                
                if len(genres) > 100:
                    genres = genres[:100] + "..."
                    
                st.markdown(f"""
                <div class='anime-card'>
                    <h4 style='margin-top: 0; color: #ffffff;'>{anime_name[:50]}{'...' if len(anime_name) > 50 else ''}</h4>
                    <p class='small-text'>🎭 <b>Genre:</b> {genres}</p>
                    <p class='rating-star'>⭐ Rating: {anime['Rating']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if total_pages > 1:
            st.markdown(f"<div style='text-align: center; color: #cccccc; margin: 1rem 0;'><b>Page {page} of {total_pages}</b></div>", unsafe_allow_html=True)
    else:
        st.warning("🚫 No anime found for the selected filters. Try adjusting your criteria!")

    # Recommendation System
    if selected_anime and selected_anime != "":
        try:
            idx = df[df["Anime"] == selected_anime].index[0]
            sim_scores = list(enumerate(similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_indices = [i for i, _ in sim_scores[1:num_results+1]]

            st.markdown("---")
            st.markdown(f"""
                <div style='text-align: center; margin: 2rem 0;'>
                    <h2>🎯 RECOMMENDATIONS FOR YOU</h2>
                    <p style='color: #cccccc;'>Anime similar to <b style='color: #4ECDC4;'>{selected_anime}</b></p>
                    <div style='height: 3px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); width: 30%; margin: 0.5rem auto; border-radius: 2px;'></div>
                </div>
            """, unsafe_allow_html=True)

            cols = st.columns(3)
            recommendation_count = 0
            for i, index in enumerate(sim_indices):
                if recommendation_count >= num_results:
                    break
                    
                anime = df.iloc[index]
                if anime["Rating"] >= min_rating:
                    with cols[recommendation_count % 3]:
                        anime_name = str(anime['Anime'])
                        genres = str(anime['Genre'])
                        
                        if len(anime_name) > 50:
                            anime_name = anime_name[:50] + "..."
                            
                        if len(genres) > 80:
                            genres = genres[:80] + "..."
                            
                        st.markdown(f"""
                        <div class='anime-card'>
                            <h4 style='margin-top: 0; color: #ffffff;'>{anime_name}</h4>
                            <p class='small-text'>🎭 {genres}</p>
                            <p class='rating-star'>⭐ {anime['Rating']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    recommendation_count += 1
                    
            if recommendation_count == 0:
                st.info("ℹ️ No recommendations found that match your minimum rating filter. Try lowering the minimum rating!")
                
        except IndexError:
            st.error("❌ Anime not found in dataset. Please select another anime.")
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {e}")

# ===============================
# 🤖 GENRE CLASSIFICATION SYSTEM
# ===============================
elif app_mode == "🤖 Genre Classification":
    
    st.markdown(f"""
        <div style='text-align: center; margin: 2rem 0;'>
            <h2>🤖 NAÏVE BAYES GENRE CLASSIFICATION</h2>
            <p style='color: #cccccc;'>Automatically classify anime into genres based on descriptions</p>
            <div style='height: 3px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); width: 30%; margin: 0.5rem auto; border-radius: 2px;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    if not classification_ready:
        st.warning("⚠️ Genre classification model is not available. Please check if there's sufficient data.")
    else:
        # Real-time classification
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Classification Demo")
            if custom_description and custom_description != "Enter anime description for genre classification...":
                try:
                    predictions, probabilities = genre_classifier.predict([custom_description])
                    predicted_genre = predictions[0]
                    prob_scores = probabilities[0]
                    
                    # Display results
                    st.markdown(f"**Predicted Genre:** `{predicted_genre}`")
                    
                    # Show top 3 probabilities
                    genre_probs = list(zip(genre_classifier.genre_labels, prob_scores))
                    genre_probs.sort(key=lambda x: x[1], reverse=True)
                    
                    st.markdown("**Top Genre Probabilities:**")
                    for genre, prob in genre_probs[:3]:
                        st.markdown(f"- `{genre}`: {prob:.2%}")
                        
                except Exception as e:
                    st.error(f"Error in classification: {e}")
        
        with col2:
            st.markdown("### ℹ️ How It Works")
            st.markdown("""
            This uses **Multinomial Naïve Bayes** to classify anime descriptions:
            
            - **TF-IDF Vectorization**: Converts text to numerical features
            - **Naïve Bayes**: Probabilistic classification algorithm
            - **Genre Prediction**: Predicts the most likely genre
            """)
        
        # Sample classifications from dataset
        st.markdown("---")
        st.markdown("### 🎯 Sample Classifications")
        
        # Get some test samples
        if classification_ready and len(X_test) > 0:
            sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
            
            for idx in sample_indices:
                actual_desc = X_test.iloc[idx]
                actual_genre = y_test.iloc[idx]
                pred_genre, probs = genre_classifier.predict([actual_desc])
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Description:** {actual_desc[:100]}...")
                with col2:
                    status = "✅" if pred_genre[0] == actual_genre else "❌"
                    st.markdown(f"**Actual:** `{actual_genre}` | **Predicted:** `{pred_genre[0]}` {status}")

# ===============================
# 📊 MODEL EVALUATION
# ===============================
elif app_mode == "📊 Model Evaluation":
    
    st.markdown(f"""
        <div style='text-align: center; margin: 2rem 0;'>
            <h2>📊 MODEL EVALUATION METRICS</h2>
            <p style='color: #cccccc;'>Performance analysis of Naïve Bayes Genre Classifier</p>
            <div style='height: 3px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); width: 30%; margin: 0.5rem auto; border-radius: 2px;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    if not classification_ready:
        st.warning("⚠️ No trained model available for evaluation.")
    else:
        try:
            # Calculate evaluation metrics
            accuracy, report, cm, predictions = genre_classifier.evaluate(X_test, y_test)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class='stats-card'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #4ECDC4;'>📊</div>
                        <div style='font-size: 0.9em;'>Accuracy</div>
                        <div style='font-size: 1.2em; font-weight: bold;'>{accuracy:.2%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                precision = report['weighted avg']['precision']
                st.markdown(f"""
                    <div class='stats-card'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #FFD93D;'>🎯</div>
                        <div style='font-size: 0.9em;'>Precision</div>
                        <div style='font-size: 1.2em; font-weight: bold;'>{precision:.2%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                recall = report['weighted avg']['recall']
                st.markdown(f"""
                    <div class='stats-card'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #FF6B6B;'>🔍</div>
                        <div style='font-size: 0.9em;'>Recall</div>
                        <div style='font-size: 1.2em; font-weight: bold;'>{recall:.2%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Confusion Matrix
            st.markdown("### 📈 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=genre_classifier.genre_labels,
                       yticklabels=genre_classifier.genre_labels,
                       ax=ax)
            ax.set_xlabel('Predicted Genre')
            ax.set_ylabel('Actual Genre')
            ax.set_title('Confusion Matrix - Genre Classification')
            st.pyplot(fig)
            
            # Classification Report
            st.markdown("### 📋 Detailed Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format({
                'precision': '{:.2%}',
                'recall': '{:.2%}', 
                'f1-score': '{:.2%}',
                'support': '{:.0f}'
            }))
            
            # Model Information
            st.markdown("### ℹ️ Model Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset Information:**")
                st.markdown(f"- Training samples: {len(X_train):,}")
                st.markdown(f"- Test samples: {len(X_test):,}")
                st.markdown(f"- Number of genres: {len(genre_classifier.genre_labels)}")
                st.markdown(f"- Feature dimensions: {genre_classifier.vectorizer.get_feature_names_out().shape[0]}")
            
            with col2:
                st.markdown("**Algorithm Details:**")
                st.markdown("- **Classifier**: Multinomial Naïve Bayes")
                st.markdown("- **Feature Extraction**: TF-IDF Vectorizer")
                st.markdown("- **Max Features**: 1,000")
                st.markdown("- **Test Split**: 30%")
                
        except Exception as e:
            st.error(f"Error in model evaluation: {e}")

# ===============================
# 👨‍💻 Footer
# ===============================
st.markdown("---")
st.markdown("""
    <div class='footer'>
        <div style='font-size: 1.1em; margin-bottom: 1rem;'>
            Built with ❤️ using <b>Streamlit</b>, <b>TF-IDF</b>, <b>Cosine Similarity</b>, and <b>Naïve Bayes</b>
        </div>
        <div class='made-by'>
            Made by Divy Jain
        </div>
        <div style='margin-top: 0.5rem; color: #888; font-size: 0.9em;'>
            Machine Learning Lab Project | Covers Lab Topic 7: Naïve Bayesian Classifier
        </div>
    </div>
""", unsafe_allow_html=True)