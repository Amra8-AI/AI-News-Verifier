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
:root {
    --bg-main: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    --glass-bg: rgba(255, 255, 255, 0.12);
    --glass-border: rgba(255, 255, 255, 0.25);
    --neon-blue: #38bdf8;
    --neon-purple: #a855f7;
    --neon-pink: #ec4899;
    --gold: #facc15;
}

/* Background */
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

/* Main container */
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
    background: linear-gradient(to right, #38bdf8, #ec4899, #facc15, #38bdf8);
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

/* Sidebar header */
[data-testid="stSidebar"] h2 {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff 30%, var(--neon-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 1rem 0;
    border-bottom: 2px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] h2::before {
    content: "";
    display: inline-block;
    width: 12px;
    height: 12px;
    background-color: var(--neon-pink);
    border-radius: 50%;
    box-shadow: 0 0 15px var(--neon-pink);
    margin-right: 12px;
    animation: pulse-glow 2s infinite;
}
@keyframes pulse-glow {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(236,72,153,0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(236,72,153,0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(236,72,153,0); }
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

/* Upload box */
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

/* Metric cards */
.metric-card {
    position: relative;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: scale(1.03);
    box-shadow: 0 12px 30px rgba(236,72,153,0.6);
}
</style>
""", unsafe_allow_html=True)

# --- LOAD OR TRAIN MODEL ---
@st.cache_resource
def load_or_train_model(uploaded_df=None):
    model_path = "news_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    if uploaded_df is None and os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        return tfidf_vectorizer, model, None, None, None, None, False

    if uploaded_df is not None:
        df = uploaded_df
    else:
        if not os.path.exists("News.csv"):
            return None, None, None, None, None, None, False
        df = pd.read_csv("News.csv")

    # Balance dataset
    df_fake = df[df.label == 'FAKE']
    df_real = df[df.label == 'REAL']
    df_real_upsampled = resample(df_real, replace=True, n_samples=len(df_fake), random_state=42)
    df_balanced = pd.concat([df_fake, df_real_upsampled])

    x_train, x_test, y_train, y_test = train_test_split(
        df_balanced['text'], df_balanced['label'], test_size=0.2, random_state=7
    )

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(tfidf_train, y_train)

    y_pred = model.predict(tfidf_vectorizer.transform(x_test))
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)

    return tfidf_vectorizer, model, y_test, y_pred, acc, cm, True

# --- MAIN APP ---
st.title("🕵️ AI News Verifier")
st.markdown("<p class='subtitle'>Upload CSV files for intelligent fake news detection with beautiful visualizations</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎓 Train Model")
uploaded_file = st.sidebar.file_uploader("📂 Upload Training CSV", type=['csv'], help="CSV must have 'text' and 'label' columns")

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    if 'text' in uploaded_df.columns and 'label' in uploaded_df.columns:
        st.sidebar.markdown(f"<div class='success-box'>✅ <strong>{len(uploaded_df)} rows</strong> loaded
