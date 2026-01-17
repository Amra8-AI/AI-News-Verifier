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
    page_title="AI News Batch Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #1e40af;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 30px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 700;
        border: none;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    .upload-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3e7f1 100%);
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #10b981;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
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
st.title("📊 AI News Batch Analyzer")
st.markdown(
    "<p class='subtitle'>Upload CSV files for intelligent fake news detection with beautiful visualizations</p>",
    unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("🎓 Train Model")
st.sidebar.markdown("Upload your training dataset to create or update the AI model")

uploaded_file = st.sidebar.file_uploader("📂 Upload Training CSV", type=['csv'],
                                         help="CSV must have 'text' and 'label' columns")

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    if 'text' in uploaded_df.columns and 'label' in uploaded_df.columns:
        st.sidebar.markdown(
            f"<div class='success-box'>✅ <strong>{len(uploaded_df)} rows</strong> loaded successfully</div>",
            unsafe_allow_html=True)
        if st.sidebar.button("🚀 Train Model"):
            with st.spinner("🔄 Training AI model..."):
                tfidf_vectorizer, model, y_test, y_pred, acc, cm, trained_fresh = load_or_train_model(uploaded_df)
                st.sidebar.success(f"✅ Model trained! Accuracy: {acc * 100:.2f}%")
                st.rerun()
    else:
        st.sidebar.error("❌ CSV must have 'text' and 'label' columns")
        uploaded_df = None
else:
    uploaded_df = None

# Load model
tfidf_vectorizer, model, y_test, y_pred, acc, cm, trained_fresh = load_or_train_model()

if model is None:
    st.error("⚠️ No model found. Please upload a training CSV file in the sidebar.")
    st.stop()

# --- SIDEBAR PERFORMANCE ---
st.sidebar.markdown("---")
st.sidebar.header("📈 Model Performance")
if trained_fresh and y_test is not None:
    st.sidebar.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #10b981; margin: 0;'>{acc * 100:.2f}%</h2>
        <p style='color: #64748b; margin: 0;'>Model Accuracy</p>
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
        plt.title("Confusion Matrix", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Predicted Label", fontsize=13, fontweight='bold')
        plt.ylabel("Actual Label", fontsize=13, fontweight='bold')
        st.sidebar.pyplot(fig)
        plt.close()

        # Download heatmap
        fig.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
        with open("confusion_matrix.png", "rb") as file:
            st.sidebar.download_button(
                label="📥 Download Heatmap",
                data=file,
                file_name="confusion_matrix.png",
                mime="image/png"
            )
else:
    st.sidebar.info("ℹ️ Model loaded from cache")

# --- MAIN BATCH ANALYSIS ---
st.markdown("---")
st.header("📂 Batch Analysis")

# Upload section
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
        <ul>
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
            st.markdown(f"<div class='success-box'><h3>✅ Loaded {len(batch_df)} articles for analysis</h3></div>",
                        unsafe_allow_html=True)

            # Analysis button
            if st.button("🚀 Analyze All Articles", use_container_width=True, key="analyze_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                predictions = []
                fake_probabilities = []
                real_probabilities = []

                for idx, row in batch_df.iterrows():
                    status_text.markdown(
                        f"<p style='text-align: center; font-size: 1.2rem;'>🔍 Analyzing article <strong>{idx + 1}/{len(batch_df)}</strong>...</p>",
                        unsafe_allow_html=True)
                    progress_bar.progress((idx + 1) / len(batch_df))

                    text = str(row['text'])

                    # Model prediction
                    tfidf_test = tfidf_vectorizer.transform([text])
                    pred = model.predict(tfidf_test)[0]
                    proba = model.predict_proba(tfidf_test)[0]

                    predictions.append(pred)
                    fake_probabilities.append(f"{proba[0]:.2%}")
                    real_probabilities.append(f"{proba[1]:.2%}")

                progress_bar.empty()
                status_text.empty()

                # Add results to dataframe
                batch_df['Prediction'] = predictions
                batch_df['FAKE_Probability'] = fake_probabilities
                batch_df['REAL_Probability'] = real_probabilities

                # Display results summary
                st.markdown("<div class='success-box'><h2>✅ Analysis Complete!</h2></div>", unsafe_allow_html=True)

                # Metrics
                col1, col2, col3 = st.columns(3)
                real_count = predictions.count('REAL')
                fake_count = predictions.count('FAKE')

                with col1:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left-color: #10b981;'>
                        <h1 style='color: #10b981; margin: 0;'>{real_count}</h1>
                        <h3 style='color: #64748b; margin: 0;'>REAL Articles</h3>
                        <p style='color: #94a3b8; font-size: 1.2rem;'>{real_count / len(predictions) * 100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left-color: #ef4444;'>
                        <h1 style='color: #ef4444; margin: 0;'>{fake_count}</h1>
                        <h3 style='color: #64748b; margin: 0;'>FAKE Articles</h3>
                        <p style='color: #94a3b8; font-size: 1.2rem;'>{fake_count / len(predictions) * 100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left-color: #667eea;'>
                        <h1 style='color: #667eea; margin: 0;'>{len(predictions)}</h1>
                        <h3 style='color: #64748b; margin: 0;'>Total Analyzed</h3>
                        <p style='color: #94a3b8; font-size: 1.2rem;'>100%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Visualizations
                st.markdown("---")
                st.subheader("📊 Visual Analytics")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Pie chart
                labels = ['REAL', 'FAKE']
                sizes = [real_count, fake_count]
                colors = ['#10b981', '#ef4444']
                explode = (0.05, 0.05)

                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                        shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
                ax1.set_title('News Classification Distribution', fontsize=18, fontweight='bold', pad=20)

                # Bar chart
                ax2.bar(labels, sizes, color=colors, edgecolor='white', linewidth=3, width=0.6)
                ax2.set_ylabel('Number of Articles', fontsize=14, fontweight='bold')
                ax2.set_title('Article Count by Category', fontsize=18, fontweight='bold', pad=20)
                ax2.grid(axis='y', alpha=0.3, linestyle='--')
                ax2.set_facecolor('#f8fafc')

                for i, v in enumerate(sizes):
                    ax2.text(i, v + max(sizes) * 0.02, str(v), ha='center', va='bottom',
                             fontweight='bold', fontsize=16)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Results table
                st.markdown("---")
                st.subheader("📋 Detailed Results")

                # Add headline column if not exists
                if 'headline' not in batch_df.columns:
                    batch_df['Headline'] = batch_df['text'].str[:100] + '...'
                    cols = ['Headline'] + [col for col in batch_df.columns if col != 'Headline']
                    batch_df = batch_df[cols]


                # Color-code the prediction column
                def highlight_prediction(val):
                    if val == 'REAL':
                        return 'background-color: #d1fae5; color: #065f46; font-weight: bold'
                    elif val == 'FAKE':
                        return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                    else:
                        return 'background-color: #fef3c7; color: #92400e; font-weight: bold'


                styled_df = batch_df.style.applymap(
                    highlight_prediction,
                    subset=['Prediction']
                )

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

                # Statistics
                with st.expander("📈 Detailed Statistics & Insights"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 📊 Summary Statistics")
                        st.write(f"**Total Articles Analyzed:** {len(batch_df)}")
                        st.write(f"**REAL Articles:** {real_count} ({real_count / len(predictions) * 100:.1f}%)")
                        st.write(f"**FAKE Articles:** {fake_count} ({fake_count / len(predictions) * 100:.1f}%)")

                    with col2:
                        st.markdown("### 🎯 Confidence Metrics")
                        st.write(
                            f"**Average FAKE Probability:** {np.mean([float(p.strip('%')) / 100 for p in fake_probabilities]):.2%}")
                        st.write(
                            f"**Average REAL Probability:** {np.mean([float(p.strip('%')) / 100 for p in real_probabilities]):.2%}")

                        # High confidence predictions
                        high_conf_fake = sum(1 for p in fake_probabilities if float(p.strip('%')) > 80)
                        high_conf_real = sum(1 for p in real_probabilities if float(p.strip('%')) > 80)
                        st.write(f"**High Confidence (>80%):** {high_conf_fake + high_conf_real} articles")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format with a 'text' column.")

else:
    # Show instructions when no file uploaded
    st.markdown("""
    <div class='upload-box'>
        <h2>👆 Upload your CSV file to get started</h2>
        <p style='font-size: 1.1rem; color: #64748b; margin-top: 1rem;'>
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
<div style='text-align: center; padding: 2rem;'>
    <h3 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               margin-bottom: 0.5rem;'>AI News Batch Analyzer</h3>
    <p style='color: #64748b;'>Powered by Machine Learning & Advanced Analytics</p>
    <p style='color: #94a3b8; font-size: 0.9rem;'>
        Upload • Analyze • Download • Repeat
    </p>
</div>
""", unsafe_allow_html=True)
