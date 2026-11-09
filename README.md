# Fake News Detector

A complete, lightweight Natural Language Processing (NLP) system that classifies news articles as **Real (0)** or **Fake (1)** using **TF-IDF vectorization** and a **Feedforward Neural Network (FFNN)** built with Keras. A **Logistic Regression** classifier is trained on identical TF-IDF features as a comparative baseline.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Environment Setup & Installation](#environment-setup--installation)
- [Dataset](#dataset)
- [Training Pipeline](#training-pipeline)
- [CLI Prediction](#cli-prediction)
- [Streamlit Web Application](#streamlit-web-application)
- [Experimental Results](#experimental-results)
- [Limitations & Viva Discussion Points](#limitations--viva-discussion-points)
- [Reproducibility](#reproducibility)

---

## Project Overview

Online misinformation spreads rapidly across media platforms. This project demonstrates how classical NLP feature extraction (TF-IDF with n-grams) combined with a compact multi-layer feedforward neural network provides high-accuracy text classification while remaining fast and interpretable.

Key Highlights:
- **Clean modular design**: All paths, hyperparameter values, and seeds live in `src/config.py`.
- **Zero data leakage**: TF-IDF vectorizer fits strictly on the training partition (70%) and transforms validation (15%) and test (15%) sets.
- **Source tag stripping**: Cleans Reuters wire-service lead-ins and image tags so the model learns from journalistic style and content rather than bylines.
- **End-to-end runnable**: From raw data loading to interactive Streamlit deployment.

---

## Folder Structure

```
fake-news-detector/
│
├── data/                         # Dataset directory
│   ├── Fake.csv                  # Kaggle fake articles (23,481 rows)
│   ├── True.csv                  # Kaggle true articles (21,417 rows)
│   └── README.txt                # Dataset source and download notes
│
├── models/                       # Serialized trained artifacts
│   ├── fake_news_nn.keras        # Saved Feedforward Neural Network model
│   └── tfidf_vectorizer.joblib   # Fitted scikit-learn TfidfVectorizer
│
├── outputs/                      # Evaluation charts and metric reports
│   ├── confusion_matrix.png      # Test set confusion matrix heatmap
│   ├── training_curves.png       # Loss and accuracy curves over epochs
│   └── metrics.json              # Recorded accuracy, precision, recall, F1
│
├── src/                          # Modular Python source code
│   ├── __init__.py
│   ├── config.py                 # Central configurations & hyperparameters
│   ├── data_loader.py            # Data ingestion, leakage stripping, shuffle
│   ├── preprocess.py             # Lowercase, regex cleaning, stopwords, lemmatization
│   ├── features.py               # TF-IDF vectorizer + batch data generator
│   ├── model.py                  # Keras Feedforward Neural Network architecture
│   ├── train.py                  # End-to-end training and evaluation pipeline
│   ├── evaluate.py               # Metrics, confusion matrix, and curve plotting
│   └── predict.py                # Single article inference + CLI interface
│
├── app.py                        # Streamlit web demo
├── requirements.txt              # Project package dependencies
└── README.md                     # Documentation and viva presentation guide
```

---

## Pipeline Architecture

1. **Data Ingestion & Leakage Cleaning (`src/data_loader.py`)**:
   - Ingests `Fake.csv` (labeled `1`) and `True.csv` (labeled `0`).
   - Strips publisher artifacts such as `"WASHINGTON (Reuters) - "` and `"(Reuters)"` from body text.
   - Combines `title` and cleaned `text` into a single `content` string.
   - Shuffles the combined 44,898 articles with a fixed random seed (`42`).

2. **Text Preprocessing (`src/preprocess.py`)**:
   - Lowercases text.
   - Removes URLs, HTML tags, punctuation, digits, and extra whitespaces.
   - Removes English stopwords (via NLTK).
   - Lemmatizes tokens using NLTK `WordNetLemmatizer`.

3. **Stratified Split (`src/train.py`)**:
   - Splits dataset into **70% Train** (31,428 articles), **15% Validation** (6,735 articles), and **15% Test** (6,735 articles) with stratification to preserve class balance.

4. **TF-IDF Feature Extraction (`src/features.py`)**:
   - Scikit-learn `TfidfVectorizer` configured with `max_features=5000`, `ngram_range=(1, 2)`, `min_df=2`, `max_df=0.9`.
   - Fitted **strictly on the training split**. Validation, test, and inference texts are transformed using this frozen vectorizer.
   - Uses a memory-efficient `SparseToDenseSequence` batch generator so large matrices do not overwhelm RAM.

5. **Neural Network Classifier (`src/model.py`)**:
   - `Input(5000)`
   - `Dense(128, activation='relu')` + `Dropout(0.5)`
   - `Dense(64, activation='relu')` + `Dropout(0.3)`
   - `Dense(1, activation='sigmoid')` (outputs probability $P(\text{Fake})$)
   - Optimizer: `Adam(learning_rate=0.001)`, Loss: `binary_crossentropy`.
   - `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`.

6. **Baseline & Evaluation (`src/evaluate.py`)**:
   - `LogisticRegression(C=1.0, max_iter=1000)` trained on identical TF-IDF representations.
   - Evaluates test accuracy, precision, recall, and F1-score for both models.

---

## Environment Setup & Installation

Requires **Python 3.10+**.

```bash
# 1. Navigate to workspace
cd "c:\Users\aisha\Desktop\fake news detector"

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset

The project uses the Kaggle **"Fake and Real News Dataset"** (Clément Bisaillon / ISOT dataset):
- Kaggle URL: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Files required: `Fake.csv` and `True.csv` placed into the `data/` folder.

Both files are already present in `data/`:
- `data/Fake.csv` (~62.8 MB, 23,481 articles)
- `data/True.csv` (~53.6 MB, 21,417 articles)

---

## Training Pipeline

To run the complete pipeline from scratch:

```bash
python -m src.train
```

Outputs generated:
- `models/fake_news_nn.keras`
- `models/tfidf_vectorizer.joblib`
- `outputs/confusion_matrix.png`
- `outputs/training_curves.png`
- `outputs/metrics.json`

---

## CLI Prediction

Classify any news article directly from the terminal:

```bash
python -m src.predict "President announces new infrastructure investments today during congressional press briefing."
```

Or pass text via stdin:

```bash
python -m src.predict < sample_article.txt
```

Example CLI Output:
```text
Prediction:        Real
Confidence:        98.02%
Fake probability:  0.0198
```

---

## Streamlit Web Application

Launch the interactive web interface:

```bash
python -m streamlit run app.py
```

*(Note: Use `python -m streamlit run app.py` if `streamlit` is not directly on your system PATH).*

Features:
- Text area to paste any news headline and body.
- Quick test buttons to load sample real or fake news articles.
- Instant classification with predicted label (`Real News` / `Fake News`).
- Probability breakdown and visual confidence bars.
- Built-in summary for viva questions.

---

## Experimental Results

Evaluated on the held-out **15% stratified test split (6,735 articles)**:

| Model | Test Accuracy | Precision (Fake) | Recall (Fake) | F1-Score (Fake) |
| :--- | :---: | :---: | :---: | :---: |
| **Feedforward Neural Network** | **99.06%** | **0.9918** | **0.9903** | **0.9910** |
| **Logistic Regression (Baseline)** | **98.29%** | **0.9869** | **0.9804** | **0.9836** |

### Confusion Matrix Breakdown (Neural Network on Test Set)
- **True Real (Correctly identified Real)**: 3,184
- **False Fake (Real misclassified as Fake)**: 29
- **False Real (Fake misclassified as Real)**: 34
- **True Fake (Correctly identified Fake)**: 3,488

Charts saved in `outputs/`:
- `outputs/confusion_matrix.png`: Visual heatmap of actual vs. predicted labels.
- `outputs/training_curves.png`: Training loss/accuracy vs. validation loss/accuracy across epochs.

---

## Limitations & Viva Discussion Points

During your project examination / viva, you should proactively address the following critical insights:

1. **Why is accuracy so high (>98%)?**
   - In the Kaggle ISOT dataset, real articles originate almost exclusively from the **Reuters news wire**, whereas fake articles are scraped from various partisan blogs and conspiracy websites.
   - Even after stripping `"(Reuters)"` lead-ins, Reuters articles follow strict journalistic style (formal grammar, inverted-pyramid reporting, wire terminology), while fake articles feature sensational vocabulary, emotional tone, and exclamation marks.
   - Consequently, the model is detecting **source genre/style differences** rather than performing factual verification against the real world.

2. **Source Leakage Mitigation**:
   - In raw `True.csv`, nearly every article opened with `"CITY (Reuters) - "`.
   - We explicitly removed these lead-in tags and residual `Reuters` tokens in `src/data_loader.py` before concatenating title and body, ensuring the model doesn't rely on publisher tags.

3. **Dropped Columns**:
   - The dataset contains `subject` (e.g., `politicsNews` vs `News`) and `date`. These columns were intentionally omitted because the category names in this dataset strongly correlate with the file source and would cause artificial label leakage.

4. **No External Fact-Checking**:
   - A TF-IDF + Neural Network model evaluates word statistics within the document; it does not query an external knowledge graph or verify whether named entities did what the text claims.

---

## Reproducibility

- `src/config.py` fixes `RANDOM_SEED = 42`.
- `set_seeds()` in `src/train.py` configures Python `random`, `numpy.random`, and `keras.utils.set_random_seed()` to ensure consistent data splitting and weight initialization.
