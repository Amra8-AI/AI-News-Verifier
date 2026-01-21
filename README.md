# AI-News-VerifierOverview: AI News Verifier is a Streamlit‑based machine learning application that detects FAKE vs REAL news articles. It leverages TF‑IDF vectorization and Logistic Regression to classify text, while offering interactive visualizations, batch analysis, and a modern UI with custom CSS styling.

Features: Upload CSV datasets to train or update the model, balance data automatically with resampling, view accuracy and confusion matrix heatmaps, analyze multiple articles in batch mode, generate pie and bar charts for classification results, explore styled tables with probability scores, and download analysis outputs in CSV format.

Tech Stack: Built with Streamlit for the interface, scikit‑learn for machine learning, pandas and numpy for data handling, matplotlib and seaborn for visualizations, and joblib for model persistence.

Requirements: The project depends on joblib, matplotlib, numpy, pandas, scikit‑learn, seaborn, and streamlit. Versions can be pinned in a requirements.txt  file to ensure reproducibility.

Usage: Clone the repository, install dependencies from requirements.txt, and run the application with Streamlit. The app launches locally in the browser, offering an interactive dashboard for training models and analyzing news datasets.

Dataset Format: CSV files must include a “text” column containing article content and a “label” column with FAKE or REAL values. An optional “headline” column can be used for display purposes.

Outputs: The app provides model accuracy, confusion matrix visualization, classification distribution charts, styled dataframes with predictions, downloadable CSV results, and detailed statistics such as confidence metrics and summary counts.

Contribution & License: Contributions are welcome through pull requests, and the project can be shared or modified under the MIT License.
