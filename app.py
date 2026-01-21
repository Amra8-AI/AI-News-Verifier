import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI News Verifier",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    /* Root Theme Variables */
    :root {
        --bg-main: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        --glass-bg: rgba(255, 255, 255, 0.12);
        --glass-border: rgba(255, 255, 255, 0.25);
        --neon-blue: #38bdf8;
        --neon-purple: #a855f7;
        --neon-pink: #ec4899;
        --gold: #facc15;
    }
    
    /* App Background */
    .stApp {
        background: var(--bg-main);
        background-attachment: fixed;
        color: #f8fafc;
        animation: aurora 15s ease infinite;
    }
    
    @keyframes aurora {
        0% {filter: hue-rotate(0deg);}
        50% {filter: hue-rotate(25deg);}
        100% {filter: hue-rotate(0deg);}
    }
    
    /* Main Container */
    .main {
        backdrop-filter: blur(18px) saturate(180%);
        background: var(--glass-bg);
        border-radius: 28px;
        padding: 2.5rem;
        margin: 1.5rem;
        border: 1px solid var(--glass-border);
        box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    }
    
    /* Headings */
    h1 {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(to right, #38bdf8 20%, #ec4899 40%, #facc15 60%, #38bdf8 80%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    h2, h3 {
        color: var(--gold);
        font-weight: 800;
        text-shadow: 0 0 8px rgba(250,204,21,0.6);
    }
    
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(120deg, var(--neon-blue), var(--neon-purple), var(--neon-pink));
        background-size: 300% 300%;
        animation: gradientMove 5s ease infinite;
        color: white;
        border-radius: 999px;
        padding: 1rem 3rem;
        font-size: 1.25rem;
        font-weight: 800;
        border: none;
        box-shadow: 0 10px 30px rgba(168,85,247,0.5);
        transition: all 0.35s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.06);
        box-shadow: 0 20px 50px rgba(236,72,153,0.8);
    }
    
    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* Upload Box */
    .upload-box {
        background: rgba(255,255,255,0.08);
        border: 2px dashed var(--neon-blue);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(56,189,248,0.6);
        transition: all 0.35s ease;
    }
    
    .upload-box:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 0 40px rgba(168,85,247,0.8);
    }
    
    /* Metric Cards */
    .metric-card {
        position: relative;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        margin: 1rem 0;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0% {transform: translateY(0px);}
        50% {transform: translateY(-6px);}
        100% {transform: translateY(0px);}
    }
    
    /* Success Box */
    .success-box {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399 !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        animation: slideInRight 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,12,41,0.95), rgba(48,43,99,0.95));
        border-right: 1px solid rgba(255,255,255,0.15);
    }
    
    [data-testid="stSidebar"] h2 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 30%, var(--neon-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem !important;
    }
    
    /* Inputs */
    input, textarea {
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
        border-radius: 16px !important;
        border: 1px solid rgba(56,189,248,0.4) !important;
        padding: 0.75rem 1rem !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--neon-purple), var(--neon-pink));
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# --- LOAD OR TRAIN MODEL ---
@st.cache_resource
def load_or_train_model(uploaded_df=None):
    """Load existing model or train new one"""
    model_path = "news_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    # Load existing model if available and no new training data
    if uploaded_df is None and os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            model = joblib.load(model_path)
            tfidf_vectorizer = joblib.load(vectorizer_path)
            return tfidf_vectorizer, model, None, None, None, None, False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None, None, None, None, None, False

    # Train new model
    if uploaded_df is not None:
        df = uploaded_df
    else:
        return None, None, None, None, None, None, False

    try:
        # Validate required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            st.error("Dataset must have 'text' and 'label' columns")
            return None, None, None, None, None, None, False

        # Clean data
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str)
        
        # Balance dataset
        df_fake = df[df.label == 'FAKE']
        df_real = df[df.label == 'REAL']
        
        if len(df_fake) == 0 or len(df_real) == 0:
            st.error("Dataset must contain both FAKE and REAL labels")
            return None, None, None, None, None, None, False
        
        df_real_upsampled = resample(df_real, replace=True, n_samples=len(df_fake), random_state=42)
        df_balanced = pd.concat([df_fake, df_real_upsampled])

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(
            df_balanced['text'], df_balanced['label'], test_size=0.2, random_state=7
        )

        # Vectorize
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)

        # Train model
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(tfidf_train, y_train)

        # Evaluate
        y_pred = model.predict(tfidf_vectorizer.transform(x_test))
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

        # Save model
        joblib.dump(model, model_path)
        joblib.dump(tfidf_vectorizer, vectorizer_path)

        return tfidf_vectorizer, model, y_test, y_pred, acc, cm, True
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None, None, False


# --- MAIN APP ---
st.title("🕵️ AI News Verifier")
st.markdown(
    "<p class='subtitle'>Upload CSV files for intelligent fake news detection with beautiful visualizations</p>",
    unsafe_allow_html=True
)

# --- SIDEBAR ---
st.sidebar.header("🎓 Train Model")
st.sidebar.markdown("Upload your training dataset to create or update the AI model")

st.sidebar.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.04);
        border-left: 3px solid var(--neon-blue);
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    ">
        <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.5; margin: 0;">
            <span style="color: var(--neon-blue); font-weight: bold;">Requirements:</span><br>
            • CSV with 'text' and 'label' columns<br>
            • Labels must be 'FAKE' or 'REAL'<br>
            • At least 100 rows recommended
        </p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Training CSV", 
    type=['csv'],
    help="CSV must have 'text' and 'label' columns"
)

uploaded_df = None
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        if 'text' in uploaded_df.columns and 'label' in uploaded_df.columns:
            st.sidebar.markdown(
                f"<div class='success-box'>✅ <strong>{len(uploaded_df)} rows</strong> loaded successfully</div>",
                unsafe_allow_html=True
            )
            if st.sidebar.button("🚀 Train Model", use_container_width=True):
                with st.spinner("🔄 Training AI model..."):
                    tfidf_vectorizer, model, y_test, y_pred, acc, cm, trained_fresh = load_or_train_model(uploaded_df)
                    if model is not None:
                        st.sidebar.success(f"✅ Model trained! Accuracy: {acc * 100:.2f}%")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Training failed. Please check your data.")
        else:
            st.sidebar.error("❌ CSV must have 'text' and 'label' columns")
            uploaded_df = None
    except Exception as e:
        st.sidebar.error(f"❌ Error reading file: {str(e)}")
        uploaded_df = None

# Load model
tfidf_vectorizer, model, y_test, y_pred, acc, cm, trained_fresh = load_or_train_model()

if model is None:
    st.warning("⚠️ No model found. Please upload a training CSV file in the sidebar to get started.")
    st.info("""
    **Getting Started:**
    1. Prepare a CSV file with two columns: 'text' (news articles) and 'label' ('FAKE' or 'REAL')
    2. Upload it using the sidebar
    3. Click 'Train Model'
    4. Once trained, you can analyze news articles!
    """)
    st.stop()

# --- SIDEBAR PERFORMANCE ---
st.sidebar.markdown("---")
st.sidebar.header("📈 Model Performance")

if trained_fresh and acc is not None:
    st.sidebar.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #10b981; margin: 0;'>{acc * 100:.2f}%</h2>
        <p style='color: #cbd5e1; margin: 0;'>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.checkbox("📊 Show Confusion Matrix", value=True):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
            xticklabels=['Predicted FAKE', 'Predicted REAL'],
            yticklabels=['Actual FAKE', 'Actual REAL'],
            cbar_kws={'label': 'Count'},
            linewidths=3,
            linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        plt.title("Confusion Matrix", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("Actual Label", fontsize=12)
        st.sidebar.pyplot(fig)
        plt.close()
else:
    st.sidebar.info("ℹ️ Model loaded from cache")

# --- MAIN BATCH ANALYSIS ---
st.markdown("---")
st.header("📂 Batch Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    batch_file = st.file_uploader(
        "📄 Upload CSV for Analysis (must have 'text' column)",
        type=['csv'],
        key='batch_analyzer',
        help="Upload a CSV file containing news articles to analyze"
    )

with col2:
    st.markdown("""
    <div class='info-box'>
        <h4>📋 Requirements</h4>
        <ul style='color: #cbd5e1;'>
            <li>CSV format</li>
            <li>'text' column required</li>
            <li>Optional: 'headline' column</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if batch_file is not None:
    try:
        batch_df = pd.read_csv(batch_file)

        if 'text' not in batch_df.columns:
            st.error("❌ CSV must contain a 'text' column with news articles")
        else:
            st.markdown(
                f"<div class='success-box'><h3>✅ Loaded {len(batch_df)} articles for analysis</h3></div>",
                unsafe_allow_html=True
            )

            if st.button("🚀 Analyze All Articles", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                predictions = []
                fake_probabilities = []
                real_probabilities = []

                for idx, row in batch_df.iterrows():
                    status_text.markdown(
                        f"<p style='text-align: center; font-size: 1.2rem; color: #cbd5e1;'>🔍 Analyzing article <strong>{idx + 1}/{len(batch_df)}</strong>...</p>",
                        unsafe_allow_html=True
                    )
                    progress_bar.progress((idx + 1) / len(batch_df))

                    text = str(row['text'])
                    tfidf_test = tfidf_vectorizer.transform([text])
                    pred = model.predict(tfidf_test)[0]
                    proba = model.predict_proba(tfidf_test)[0]

                    predictions.append(pred)
                    fake_probabilities.append(f"{proba[0]:.2%}")
                    real_probabilities.append(f"{proba[1]:.2%}")

                progress_bar.empty()
                status_text.empty()

                # Add results
                batch_df['Prediction'] = predictions
                batch_df['FAKE_Probability'] = fake_probabilities
                batch_df['REAL_Probability'] = real_probabilities

                st.markdown("<div class='success-box'><h2>✅ Analysis Complete!</h2></div>", unsafe_allow_html=True)

                # Metrics
                col1, col2, col3 = st.columns(3)
                real_count = predictions.count('REAL')
                fake_count = predictions.count('FAKE')

                with col1:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left: 5px solid #10b981;'>
                        <h1 style='color: #10b981; margin: 0;'>{real_count}</h1>
                        <h3 style='color: #cbd5e1; margin: 0;'>REAL Articles</h3>
                        <p style='color: #94a3b8; font-size: 1.2rem;'>{real_count / len(predictions) * 100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left: 5px solid #ef4444;'>
                        <h1 style='color: #ef4444; margin: 0;'>{fake_count}</h1>
                        <h3 style='color: #cbd5e1; margin: 0;'>FAKE Articles</h3>
                        <p style='color: #94a3b8; font-size: 1.2rem;'>{fake_count / len(predictions) * 100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left: 5px solid #667eea;'>
                        <h1 style='color: #667eea; margin: 0;'>{len(predictions)}</h1>
                        <h3 style='color: #cbd5e1; margin: 0;'>Total Analyzed</h3>
                        <p style='color: #94a3b8; font-size: 1.2rem;'>100%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Visualizations
                st.markdown("---")
                st.subheader("📊 Visual Analytics")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                fig.patch.set_facecolor('#1a1a2e')

                # Pie chart
                labels = ['REAL', 'FAKE']
                sizes = [real_count, fake_count]
                colors = ['#10b981', '#ef4444']
                explode = (0.05, 0.05)

                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                        shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'})
                ax1.set_title('News Classification Distribution', fontsize=18, fontweight='bold', pad=20, color='white')
                ax1.set_facecolor('#1a1a2e')

                # Bar chart
                ax2.bar(labels, sizes, color=colors, edgecolor='white', linewidth=3, width=0.6)
                ax2.set_ylabel('Number of Articles', fontsize=14, fontweight='bold', color='white')
                ax2.set_title('Article Count by Category', fontsize=18, fontweight='bold', pad=20, color='white')
                ax2.grid(axis='y', alpha=0.3, linestyle='--', color='white')
                ax2.set_facecolor('#1a1a2e')
                ax2.tick_params(colors='white')

                for i, v in enumerate(sizes):
                    ax2.text(i, v + max(sizes) * 0.02, str(v), ha='center', va='bottom',
                             fontweight='bold', fontsize=16, color='white')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Results table
                st.markdown("---")
                st.subheader("📋 Detailed Results")

                if 'headline' not in batch_df.columns:
                    batch_df['Headline'] = batch_df['text'].str[:100] + '...'
                
                def highlight_prediction(val):
                    if val == 'REAL':
                        return 'background-color: #d1fae5; color: #065f46; font-weight: bold'
                    elif val == 'FAKE':
                        return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                    return ''

                styled_df = batch_df.style.map(highlight_prediction, subset=['Prediction'])
                st.dataframe(styled_df, use_container_width=True, height=500)

                # Download results
                st.markdown("---")
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis Results (CSV)",
                    data=csv,
                    file_name="news_analysis_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format with a 'text' column.")
else:
    st.markdown("""
    <div class='upload-box'>
        <h2>👆 Upload your CSV file to get started</h2>
        <p style='font-size: 1.1rem; color: #cbd5e1; margin-top: 1rem;'>
            Drag and drop your CSV file above or click to browse
        </p>
        <p style='color: #94a3b8; margin-top: 1rem;'>
            Supported format: CSV with 'text' column containing news articles
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 2rem;'>
    <p style='font-size: 0.9rem;'>🕵️ AI News Verifier • Powered by Machine Learning</p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem;'>Built with Streamlit & scikit-learn</p>
</div>
""", unsafe_allow_html=True)
