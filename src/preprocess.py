"""
Text cleaning for news articles.

Steps (in order):
  1. Lowercase
  2. Remove URLs and HTML tags
  3. Remove punctuation and numbers
  4. Collapse extra whitespace
  5. Remove English stopwords
  6. Lemmatize (reduce words to a dictionary form: "reported" -> "report")

NLTK data is downloaded automatically the first time this module is imported.
"""

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"<.*?>")
_NON_LETTERS_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _ensure_nltk_data() -> None:
    """Download only the NLTK resources we need, quietly, if they are missing."""
    packages = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for package, path in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()


@lru_cache(maxsize=1)
def _stopwords() -> set[str]:
    return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


@lru_cache(maxsize=50000)
def _lemmatize_word(word: str) -> str:
    """Cached lemmatizer so repeated words across news articles resolve in nanoseconds."""
    return _lemmatizer().lemmatize(word)


def clean_text(text: str) -> str:
    """Clean one article. Returns a string of lemmas separated by spaces."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_LETTERS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    stop = _stopwords()
    tokens = [_lemmatize_word(token) for token in text.split() if token not in stop]
    return " ".join(tokens)


# Pre-warm lemmatizer on module load so WordNet's ~4s disk load occurs at startup, not during prediction
try:
    _lemmatize_word("warmup")
except Exception:
    pass


def clean_corpus(texts) -> list[str]:
    """Clean a list / array of articles. Prints progress every 5,000 rows."""
    cleaned = []
    total = len(texts)
    for i, text in enumerate(texts, start=1):
        cleaned.append(clean_text(text))
        if i % 5000 == 0 or i == total:
            print(f"  Preprocessed {i:,}/{total:,} articles")
    return cleaned
