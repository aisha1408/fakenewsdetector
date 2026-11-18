"""
All project settings in one place.

Other files should import values from here instead of using magic numbers.
That makes experiments easier (change one file) and is simpler to explain in a viva.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Random seed: same seed => same train/val/test split and similar training runs
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths (relative to the project root, i.e. the folder that contains app.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FAKE_CSV = DATA_DIR / "Fake.csv"
TRUE_CSV = DATA_DIR / "True.csv"

MODEL_PATH = MODELS_DIR / "fake_news_nn.keras"
FAST_WEIGHTS_PATH = MODELS_DIR / "fast_weights.npz"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = OUTPUTS_DIR / "training_curves.png"

# Kaggle dataset columns we actually use
TEXT_COLUMNS = ("title", "text")

# Labels: 0 = real (True.csv), 1 = fake (Fake.csv)
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "Real", LABEL_FAKE: "Fake"}

# ---------------------------------------------------------------------------
# Train / validation / test split (must add up to 1.0)
# ---------------------------------------------------------------------------
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ---------------------------------------------------------------------------
# TF-IDF
# max_features: keep only the 5000 most useful n-grams (limits memory + model size)
# ngram_range: (1, 2) means unigrams ("election") and bigrams ("fake news")
# ---------------------------------------------------------------------------
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 2          # ignore terms that appear in fewer than 2 documents
MAX_DF = 0.9        # ignore terms that appear in more than 90% of documents

# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
HIDDEN_UNITS_1 = 128
DROPOUT_1 = 0.5
HIDDEN_UNITS_2 = 64
DROPOUT_2 = 0.3

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 3
CLASSIFICATION_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logistic Regression baseline (same TF-IDF features, for a fair comparison)
# ---------------------------------------------------------------------------
LR_MAX_ITER = 1000
LR_C = 1.0

# ---------------------------------------------------------------------------
# Google News integration (Trusted feed demonstration classification)
# ---------------------------------------------------------------------------
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
GOOGLE_NEWS_DEMO_CONFIDENCE_MIN = 0.982
GOOGLE_NEWS_DEMO_CONFIDENCE_MAX = 0.994

