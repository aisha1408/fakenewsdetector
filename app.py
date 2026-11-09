"""
Streamlit demo for the fake news detector.

Run from the project root:
    streamlit run app.py

The app loads the files saved by `python -m src.train`.
If those files are missing, it shows setup instructions instead of crashing.
"""

import streamlit as st

from src.config import CLASSIFICATION_THRESHOLD, MODEL_PATH, VECTORIZER_PATH
from src.predict import predict

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

st.title("📰 Fake News Detector")
st.caption(
    "TF-IDF feature extraction + Feedforward Neural Network. "
    "Classroom demo project evaluating real vs. fake political news articles."
)


@st.cache_resource
def artifacts_exist() -> bool:
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()


if not artifacts_exist():
    st.error("The trained model is not available yet.")
    st.markdown(
        """
Train it first, then restart this app:

```bash
python -m src.train
python -m streamlit run app.py
```

Make sure `data/Fake.csv` and `data/True.csv` are in place before training.
"""
    )
    st.stop()

# Quick sample loaders for viva demonstration
st.write("**Quick Test Samples:**")
sample_col1, sample_col2 = st.columns(2)
with sample_col1:
    if st.button("Load Real News Sample", use_container_width=True):
        st.session_state["article_input"] = (
            "Federal Reserve officials signaled a cautious approach to future interest rate "
            "adjustments following their monthly monetary policy meeting. Economic indicators "
            "point to continued employment growth while core inflation rates continue to moderate."
        )
with sample_col2:
    if st.button("Load Fake News Sample", use_container_width=True):
        st.session_state["article_input"] = (
            "SHOCKING BOMBSHELL: Uncovered secret documents reveal massive global government "
            "coverup that mainstream media refuses to broadcast! Share this viral report immediately "
            "before global elites take it down!"
        )

article = st.text_area(
    "Paste a news article (headline + body works best)",
    value=st.session_state.get("article_input", ""),
    height=220,
    placeholder="Paste the full article text here...",
)

col1, col2 = st.columns([1, 2])
with col1:
    classify = st.button("Classify Article", type="primary", use_container_width=True)

if classify:
    try:
        result = predict(article)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        label = result["label"]
        confidence = result["confidence"]
        fake_p = result["fake_probability"]

        if label == "Fake":
            st.error(f"Prediction: **Fake News** ❌")
        else:
            st.success(f"Prediction: **Real News** ✅")

        st.write(f"Confidence in prediction: **{confidence * 100:.2f}%**")
        st.progress(min(max(confidence, 0.0), 1.0))

        st.write(f"Model output P(Fake): **{fake_p:.4f}**")
        st.progress(min(max(fake_p, 0.0), 1.0))
        st.caption(
            f"Classification decision boundary: P(Fake) ≥ {CLASSIFICATION_THRESHOLD:.2f} => Fake, "
            f"else Real."
        )

st.divider()
st.markdown(
    """
### 🎓 How it works (Viva Summary)
1. **Source Leakage Removal**: Strips wire agency tags (e.g. `(Reuters)`) from article text so the model learns genuine linguistic patterns rather than source bylines.
2. **Text Preprocessing**: Lowercases, strips URLs/HTML/punctuation/numbers, removes stopwords, and lemmatizes tokens with NLTK WordNet.
3. **TF-IDF Extraction**: Top 5,000 unigram and bigram features fitted on training data only (preventing test set leakage).
4. **Feedforward Neural Network**: Multi-layer perceptron `5000 -> Dense(128) -> Dropout(0.5) -> Dense(64) -> Dropout(0.3) -> Dense(1, sigmoid)`.
5. **Evaluation**: Evaluated on an independent 15% test set alongside a Logistic Regression baseline.
"""
)
