# predictor.py

import joblib
from pathlib import Path
from scipy.sparse import hstack


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Final combined category model
MODEL_FILE = MODEL_DIR / "category_final_model.joblib"
CHAR_VECTORIZER_FILE = MODEL_DIR / "category_final_char_vectorizer.joblib"
WORD_VECTORIZER_FILE = MODEL_DIR / "category_final_word_vectorizer.joblib"

# Category Random Forest baseline
CATEGORY_RF_VECTORIZER_FILE = MODEL_DIR / "category_rf_vectorizer.joblib"
CATEGORY_RF_MODEL_FILE = MODEL_DIR / "category_rf_model.joblib"

# Urgency Random Forest
URGENCY_RF_VECTORIZER_FILE = MODEL_DIR / "urgency_rf_vectorizer.joblib"
URGENCY_RF_MODEL_FILE = MODEL_DIR / "urgency_rf_model.joblib"


# ============================================================
# LOAD MODELS + VECTORIZERS (once, at import time)
# ============================================================

# Final combined category model
_model = joblib.load(MODEL_FILE)
_char_vectorizer = joblib.load(CHAR_VECTORIZER_FILE)
_word_vectorizer = joblib.load(WORD_VECTORIZER_FILE)

# Category RF baseline
_category_rf_vectorizer = joblib.load(CATEGORY_RF_VECTORIZER_FILE)
_category_rf_model = joblib.load(CATEGORY_RF_MODEL_FILE)

# Urgency RF
_urgency_rf_vectorizer = joblib.load(URGENCY_RF_VECTORIZER_FILE)
_urgency_rf_model = joblib.load(URGENCY_RF_MODEL_FILE)


# ============================================================
# PREDICT: FINAL CATEGORY (char + word combined)
# ============================================================

def predict_category(text: str) -> dict:
    """
    Predict the category of a ticket text using the combined
    char + word TF-IDF model.

    Args:
        text (str): The raw ticket text.

    Returns:
        dict: {"category": str, "confidence": float}
    """
    # 1. Character TF-IDF (use .transform(), NEVER .fit_transform())
    char_features = _char_vectorizer.transform([text])

    # 2. Word TF-IDF
    word_features = _word_vectorizer.transform([text])

    # 3. Combine exactly as done during training
    combined_features = hstack([char_features, word_features])

    # 4. Predict
    prediction = _model.predict(combined_features)[0]

    # 5. Confidence
    probabilities = _model.predict_proba(combined_features)[0]
    confidence = float(probabilities.max())

    return {
        "category": str(prediction),
        "confidence": round(confidence, 4),
    }


# ============================================================
# PREDICT: CATEGORY RF BASELINE (word TF-IDF only)
# ============================================================

def predict_category_rf(text: str) -> dict:
    """
    Predict the category of a ticket text using the Random Forest
    baseline model (word TF-IDF only).

    Args:
        text (str): The raw ticket text.

    Returns:
        dict: {"category": str, "confidence": float}
    """
    # 1. Word TF-IDF
    features = _category_rf_vectorizer.transform([text])

    # 2. Predict
    prediction = _category_rf_model.predict(features)[0]

    # 3. Confidence
    probabilities = _category_rf_model.predict_proba(features)[0]
    confidence = float(probabilities.max())

    return {
        "category": str(prediction),
        "confidence": round(confidence, 4),
    }


# ============================================================
# PREDICT: URGENCY RF (word TF-IDF only)
# ============================================================

def predict_urgency_rf(text: str) -> dict:
    """
    Predict the urgency of a ticket text using the Random Forest
    urgency model (word TF-IDF only).

    Args:
        text (str): The raw ticket text.

    Returns:
        dict: {"urgency": str, "confidence": float}
    """
    # 1. Word TF-IDF
    features = _urgency_rf_vectorizer.transform([text])

    # 2. Predict
    prediction = _urgency_rf_model.predict(features)[0]

    # 3. Confidence
    probabilities = _urgency_rf_model.predict_proba(features)[0]
    confidence = float(probabilities.max())

    return {
        "urgency": str(prediction),
        "confidence": round(confidence, 4),
    }