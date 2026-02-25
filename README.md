# URL Phishing Detection System

A robust, machine learning-powered URL phishing detection platform built with Python, Scikit-Learn, and Streamlit. This application uses a custom ensemble of Support Vector Machine (SVM), Logistic Regression, and Naive Bayes classifiers to identify malicious domains based on 48 lexical and structural features.

## Features
- **Real-Time Classification**: Instant analysis of any URL against a pre-trained ML ensemble.
- **Batch Processing**: Upload CSV files for automated bulk classification.
- **Trusted Domains Bypass**: Built-in whitelist for major authentic domains (Google, Apple, Microsoft) to prevent false positives.
- **Risk-Coverage Tuner**: Interactive sliders to adjust rejection thresholds dynamically.
- **Model Explainability**: Displays top 15 features by absolute weight and interactive Confusion Matrices.
- **Modern UI**: Custom glassmorphism aesthetic with dark mode integration.

## Project Structure
```text
phishing-detector/
├── app.py                      # Main Streamlit application entry point
├── train_models.py             # Model training script
├── requirements.txt            # Python dependencies
├── .gitignore
├── ml_models/                  # Trained Joblib binaries and performance metrics
├── data/                       # CSV dataset for training model
├── services/
│   ├── classification.py       # ML Pipeline routing and Trusted Domain logic
│   └── feature_extraction.py   # Lexical URL parser
└── components/                 # Reusable Streamlit UI components
```

## Quick Start
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Re-train the models:**
   The `ml_models` directory already contains pre-trained models. If you want to train them from scratch against the dataset:
   ```bash
   python train_models.py
   ```

4. **Launch the Application:**
   ```bash
   python -m streamlit run app.py
   ```

## Built With
* [Streamlit](https://streamlit.io/) - The web framework used
* [Scikit-Learn](https://scikit-learn.org/) - Machine Learning algorithms
* [Plotly](https://plotly.com/) - Interactive data visualization
* [Pandas](https://pandas.pydata.org/) - Data manipulation

## License
This project is open-sourced under the MIT License.
