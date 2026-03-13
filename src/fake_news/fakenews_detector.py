"""
Fake News Detection Module
Author: [Your Name]
Description: Detects fake news using DistilBERT transformer + TF-IDF Voting Ensemble
Trained on: ISOT, FakeReal, WELFake, Jainpooja datasets (63,590 samples)
Average Accuracy: 95.9% (DistilBERT), 94.7% (TF-IDF Ensemble)
"""

import re
import pickle
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from scipy.sparse import hstack, csr_matrix

# ── Paths to model files (adjust if needed) ──────────────────────────────────
TFIDF_PATH       = "tfidf_vectorizer_final.pkl"
LR_MODEL_PATH    = "lr_model_final.pkl"
DISTILBERT_PATH  = "distilbert_model_final"   # folder

# ── Load models once at import time ──────────────────────────────────────────
_tfidf      = None
_lr_model   = None
_tokenizer  = None
_db_model   = None


def _load_models():
    global _tfidf, _lr_model, _tokenizer, _db_model
    if _tfidf is None:
        with open(TFIDF_PATH, "rb") as f:
            _tfidf = pickle.load(f)
        with open(LR_MODEL_PATH, "rb") as f:
            _lr_model = pickle.load(f)
    if _tokenizer is None:
        _tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
        _db_model  = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
        _db_model.eval()


# ── Internal helpers ──────────────────────────────────────────────────────────
def _clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_features(texts):
    features = []
    for text in texts:
        original = str(text)
        caps_ratio    = sum(1 for c in original if c.isupper()) / (len(original) + 1)
        exclaim       = original.count("!")
        question      = original.count("?")
        caps_words    = sum(1 for w in original.split() if w.isupper() and len(w) > 2)
        word_count    = len(original.split())
        avg_word_len  = np.mean([len(w) for w in original.split()]) if original.split() else 0
        sensational   = sum(1 for w in [
                            "breaking", "shocking", "urgent", "exposed", "bombshell",
                            "secret", "leaked", "confirmed", "exclusive", "warning",
                            "must share", "hidden truth", "wake up", "banned",
                            "suppressed", "cover up", "mainstream media", "they dont want"
                         ] if w in original.lower())
        numeric_count = len(re.findall(r'\d+', original))
        quote_count   = original.count('"') + original.count("'")
        has_url       = 1 if re.search(r'http\S+|www\S+', original) else 0
        ellipsis      = original.count("...")
        has_source    = 1 if any(w in original.lower() for w in [
                            "reuters", "associated press", "according to", "published",
                            "study", "research", "percent", "university", "institute",
                            "journal", "report"]) else 0
        has_quote     = 1 if '"' in original or "said" in original.lower() else 0
        features.append([caps_ratio, exclaim, question, caps_words,
                         word_count, avg_word_len, sensational,
                         numeric_count, quote_count, has_url, ellipsis,
                         has_source, has_quote])
    return csr_matrix(np.array(features))


def _predict_tfidf(text):
    cleaned = _clean_text(text)
    tfidf_feat = _tfidf.transform([cleaned])
    extra_feat = _extract_features([text])
    combined   = hstack([tfidf_feat, extra_feat])
    pred  = _lr_model.predict(combined)[0]
    probs = _lr_model.predict_proba(combined)[0]
    return pred, probs


def _predict_distilbert(text):
    cleaned = _clean_text(text)
    inputs  = _tokenizer(cleaned, truncation=True, padding=True,
                         max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = _db_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
    pred  = int(np.argmax(probs))
    return pred, probs


# ── MAIN FUNCTION (required by dashboard) ────────────────────────────────────
def detect_fake_news(text: str) -> dict:
    """
    Detect whether a news article is fake or real.

    Args:
        text (str): The news article text to analyze.

    Returns:
        dict: {
            'is_fake'        : bool,
            'confidence'     : float (0-100),
            'verdict'        : 'FAKE' or 'REAL' or 'UNCERTAIN',
            'distilbert_real': float,
            'distilbert_fake': float,
            'tfidf_real'     : float,
            'tfidf_fake'     : float,
            'warnings'       : list of str
        }
    """
    _load_models()

    if not text or not text.strip():
        return {
            "is_fake": None, "confidence": 0.0,
            "verdict": "UNCERTAIN", "error": "Empty text provided"
        }

    # ── Run both models ───────────────────────────────────────────────────────
    db_pred, db_probs  = _predict_distilbert(text)
    lr_pred, lr_probs  = _predict_tfidf(text)

    db_real = float(db_probs[1])
    db_fake = float(db_probs[0])
    lr_real = float(lr_probs[1])
    lr_fake = float(lr_probs[0])

    # ── Weighted ensemble (70% DistilBERT + 30% TF-IDF) ──────────────────────
    combined_real = db_real * 0.7 + lr_real * 0.3
    combined_fake = db_fake * 0.7 + lr_fake * 0.3

    # ── Verdict ───────────────────────────────────────────────────────────────
    if combined_real > 0.65:
        verdict    = "REAL"
        is_fake    = False
        confidence = round(combined_real * 100, 2)
    elif combined_fake > 0.65:
        verdict    = "FAKE"
        is_fake    = True
        confidence = round(combined_fake * 100, 2)
    else:
        verdict    = "UNCERTAIN"
        is_fake    = None
        confidence = round(max(combined_real, combined_fake) * 100, 2)

    # ── Warning signals ───────────────────────────────────────────────────────
    warnings = []
    if sum(1 for c in text if c.isupper()) / (len(text) + 1) > 0.15:
        warnings.append("Excessive capital letters")
    if text.count("!") > 2:
        warnings.append(f"Too many exclamation marks ({text.count('!')})")
    sensational = ["breaking", "shocking", "urgent", "exposed", "bombshell",
                   "secret", "leaked", "exclusive", "warning", "must share",
                   "suppressed", "cover up", "wake up", "banned"]
    found = [w for w in sensational if w in text.lower()]
    if found:
        warnings.append(f"Sensational language detected: {', '.join(found[:3])}")
    if not any(c.isdigit() for c in text):
        warnings.append("No statistics or numbers found")
    credible = ["reuters", "associated press", "according to", "study",
                "research", "published", "percent", "university", "journal"]
    if not any(w in text.lower() for w in credible):
        warnings.append("No credible source indicators found")

    return {
        "is_fake"         : is_fake,
        "confidence"      : confidence,
        "verdict"         : verdict,
        "distilbert_real" : round(db_real * 100, 2),
        "distilbert_fake" : round(db_fake * 100, 2),
        "tfidf_real"      : round(lr_real * 100, 2),
        "tfidf_fake"      : round(lr_fake * 100, 2),
        "warnings"        : warnings
    }


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    fake_text = """BREAKING BOMBSHELL: Scientists CONFIRM 5G towers are secretly
    mind control devices installed by globalist elites. The mainstream media is
    BANNED from reporting this shocking truth. MUST SHARE before they delete this!"""

    real_text = """The Federal Reserve raised interest rates by 25 basis points,
    the tenth consecutive increase since March 2022. The decision was unanimous
    among the twelve voting members of the Federal Open Market Committee according
    to the official statement released on Wednesday."""

    print("Testing Fake News Detector...")
    print("-" * 50)

    result1 = detect_fake_news(fake_text)
    print(f"Test 1 - Expected FAKE")
    print(f"Verdict    : {result1['verdict']}")
    print(f"Confidence : {result1['confidence']}%")
    print(f"Warnings   : {result1['warnings']}")
    print()

    result2 = detect_fake_news(real_text)
    print(f"Test 2 - Expected REAL")
    print(f"Verdict    : {result2['verdict']}")
    print(f"Confidence : {result2['confidence']}%")
    print(f"Warnings   : {result2['warnings']}")
