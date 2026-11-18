"""
Streamlit application for Fake News Detection.

Features:
1. Google News Integration:
   - Fetches live top headlines dynamically from Google News RSS.
   - Demonstrates trusted publisher classification (marked as TRUE with high confidence 98–99%).
   - Bypasses the offline model for this verified feed.
2. Custom Article Detector:
   - Evaluates user-submitted articles/headlines using the trained
     TF-IDF + Feedforward Neural Network model.
"""

from __future__ import annotations

import streamlit as st

from src.config import (
    CLASSIFICATION_THRESHOLD,
    FAST_WEIGHTS_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
)
from src.google_news import fetch_google_news
from src.predict import predict, warmup

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Fake News Detection System")
st.caption(
    "Automated text classification using TF-IDF feature extraction, a Feedforward Neural Network, "
    "and live Google News integration."
)


@st.cache_resource
def artifacts_exist() -> bool:
    has_model = FAST_WEIGHTS_PATH.exists() or MODEL_PATH.exists()
    return has_model and VECTORIZER_PATH.exists()


@st.cache_resource(show_spinner="Initializing prediction engine...")
def initialize_engine() -> bool:
    warmup()
    return True


if not artifacts_exist():
    st.error("The trained model artifacts were not found in the models/ directory.")
    st.markdown(
        """
Please run the training pipeline first to generate the model and vectorizer:

```bash
python -m src.train
python -m streamlit run app.py
```
"""
    )
    st.stop()

# Pre-warm vectorizer and weights for instant sub-millisecond custom article prediction
initialize_engine()


@st.cache_data(ttl=180, show_spinner=False)
def load_live_news(limit: int = 15) -> list[dict]:
    return fetch_google_news(limit=limit)


tab1, tab2 = st.tabs(["🌐 Live Google News (Trusted Feed)", "🔍 Custom Article Detector"])

# -----------------------------------------------------------------------------
# TAB 1: Live Google News Integration
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Today's Headlines from Google News")
    st.markdown(
        "Real-time news headlines dynamically fetched from the **Google News** RSS feed. "
        "Under project demonstration rules, headlines indexed by Google News from established "
        "publishers are designated as **TRUE** with high confidence without routing through the offline classifier."
    )

    col_ctrl1, col_ctrl2 = st.columns([1, 4])
    with col_ctrl1:
        refresh = st.button("🔄 Refresh Headlines", use_container_width=True)
        if refresh:
            st.cache_data.clear()

    with st.spinner("Fetching latest headlines from Google News..."):
        try:
            headlines = load_live_news(limit=15)
        except Exception as err:
            st.error(f"Unable to load Google News feed: {err}")
            headlines = []

    if headlines:
        st.caption(f"Showing **{len(headlines)}** live headlines.")
        for item in headlines:
            with st.container(border=True):
                col_info, col_status = st.columns([3.8, 1.2])

                with col_info:
                    st.markdown(f"#### {item['headline']}")
                    st.markdown(
                        f"🏛️ **Source:** `{item['source']}` &nbsp;|&nbsp; "
                        f"📅 **Published:** {item['pub_date']}"
                    )
                    st.markdown(f"[🔗 Open original article on Google News]({item['link']})")

                with col_status:
                    # Green TRUE badge
                    st.markdown(
                        """
                        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; 
                                    color: #155724; padding: 6px 14px; border-radius: 8px; 
                                    text-align: center; font-weight: 700; font-size: 1.1em;
                                    margin-bottom: 6px;">
                            ✅ TRUE
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    conf_pct = item["confidence"] * 100
                    st.write(f"Confidence: **{conf_pct:.2f}%**")
                    st.progress(item["confidence"])
                    st.caption("Trusted source override")
    else:
        st.warning("No headlines available right now. Please click 'Refresh Headlines' to try again.")


# -----------------------------------------------------------------------------
# TAB 2: Custom / Manual News Classifier (Original Trained Model)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Manual Article / Headline Verification")
    st.markdown(
        "Evaluate any custom headline or news article using the original trained "
        "**TF-IDF + Feedforward Neural Network** model."
    )

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
        "Paste a news article or headline here:",
        value=st.session_state.get("article_input", ""),
        height=200,
        placeholder="Paste article headline and body text here...",
    )

    btn_col, _ = st.columns([1, 2])
    with btn_col:
        classify = st.button("Classify Article", type="primary", use_container_width=True)

    if classify:
        if not article.strip():
            st.warning("Please enter or paste news text to classify.")
        else:
            try:
                result = predict(article)
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
            else:
                label = result["label"]
                confidence = result["confidence"]
                fake_p = result["fake_probability"]

                if label == "Fake":
                    st.error("Prediction: **Fake News** ❌")
                else:
                    st.success("Prediction: **Real News** ✅")

                st.write(f"Confidence in prediction: **{confidence * 100:.2f}%**")
                st.progress(min(max(confidence, 0.0), 1.0))

                st.write(f"Model output P(Fake): **{fake_p:.4f}**")
                st.progress(min(max(fake_p, 0.0), 1.0))
                if result.get("source_type"):
                    st.caption(f"Verification method: **{result['source_type']}**")
                st.caption(
                    f"Decision boundary: P(Fake) ≥ {CLASSIFICATION_THRESHOLD:.2f} => Fake, "
                    f"else Real."
                )
