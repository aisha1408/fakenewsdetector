"""
Google News integration module.

Fetches today's live headlines from the official Google News RSS feed.
Under project demonstration rules, headlines from this trusted feed
are automatically classified as TRUE / REAL with high confidence (98–99%)
and bypass the offline TF-IDF + Neural Network model.
"""

from __future__ import annotations

import email.utils
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

from src.config import (
    GOOGLE_NEWS_DEMO_CONFIDENCE_MAX,
    GOOGLE_NEWS_DEMO_CONFIDENCE_MIN,
    GOOGLE_NEWS_RSS_URL,
)


def _compute_demo_confidence(headline: str) -> float:
    """Generate a consistent high confidence (98.0% - 99.4%) based on title hash."""
    h = int(hashlib.md5(headline.encode("utf-8")).hexdigest()[:6], 16)
    span = GOOGLE_NEWS_DEMO_CONFIDENCE_MAX - GOOGLE_NEWS_DEMO_CONFIDENCE_MIN
    return round(GOOGLE_NEWS_DEMO_CONFIDENCE_MIN + (h % 1000) / 1000.0 * span, 4)


def fetch_google_news(limit: int = 15) -> list[dict]:
    """
    Fetch top headlines from Google News RSS feed.

    Returns a list of dictionaries with:
      - headline: Cleaned news title
      - source: News publisher (e.g., BBC, The Washington Post)
      - pub_date: Publication date string formatted for display
      - link: Direct URL to the article on Google News
      - label: "TRUE" (trusted source demonstration classification)
      - confidence: High confidence value (e.g., 0.985 to 0.994)
    """
    req = urllib.request.Request(
        GOOGLE_NEWS_RSS_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
    except Exception as exc:
        raise RuntimeError(f"Could not fetch Google News feed: {exc}") from exc

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse Google News XML feed: {exc}") from exc

    articles: list[dict] = []
    items = root.findall("./channel/item")

    for item in items[:limit]:
        raw_title = item.findtext("title") or "Untitled Headline"
        link = item.findtext("link") or "#"
        raw_pub_date = item.findtext("pubDate") or ""

        # Extract publisher source
        source_elem = item.find("source")
        if source_elem is not None and source_elem.text:
            source = source_elem.text.strip()
        else:
            source = "Google News"

        # Separate source from title if embedded at the end ("Headline - Source")
        headline = raw_title.strip()
        if " - " in headline:
            parts = headline.rsplit(" - ", 1)
            headline = parts[0].strip()
            if source == "Google News" and len(parts) > 1 and parts[1].strip():
                source = parts[1].strip()

        # Format publication date nicely if available
        formatted_date = raw_pub_date
        if raw_pub_date:
            try:
                dt = email.utils.parsedate_to_datetime(raw_pub_date)
                formatted_date = dt.strftime("%b %d, %Y • %I:%M %p %Z")
            except Exception:
                formatted_date = raw_pub_date

        confidence = _compute_demo_confidence(headline)

        articles.append(
            {
                "headline": headline,
                "source": source,
                "pub_date": formatted_date,
                "link": link,
                "label": "TRUE",
                "confidence": confidence,
            }
        )

    return articles
