import joblib
from pathlib import Path
from scipy.sparse import hstack


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# Department — Logistic Regression (char + word TF-IDF)
DEPT_LOGREG_MODEL = MODEL_DIR / "category_final_model.joblib"
DEPT_LOGREG_CHAR = MODEL_DIR / "category_final_char_vectorizer.joblib"
DEPT_LOGREG_WORD = MODEL_DIR / "category_final_word_vectorizer.joblib"

# Department — Random Forest (word TF-IDF)
DEPT_RF_MODEL = MODEL_DIR / "category_rf_model.joblib"
DEPT_RF_VEC = MODEL_DIR / "category_rf_vectorizer.joblib"

# Urgency — Logistic Regression (char + word TF-IDF)
URG_LOGREG_MODEL = MODEL_DIR / "urgency_combined_model.joblib"
URG_LOGREG_CHAR = MODEL_DIR / "urgency_combined_char_vectorizer.joblib"
URG_LOGREG_WORD = MODEL_DIR / "urgency_combined_word_vectorizer.joblib"

# Urgency — Random Forest (word TF-IDF)
URG_RF_MODEL = MODEL_DIR / "urgency_rf_model.joblib"
URG_RF_VEC = MODEL_DIR / "urgency_rf_vectorizer.joblib"


# ============================================================
# LOAD MODELS
# ============================================================

dept_logreg_model = joblib.load(DEPT_LOGREG_MODEL)
dept_logreg_char = joblib.load(DEPT_LOGREG_CHAR)
dept_logreg_word = joblib.load(DEPT_LOGREG_WORD)

dept_rf_model = joblib.load(DEPT_RF_MODEL)
dept_rf_vec = joblib.load(DEPT_RF_VEC)

urg_logreg_model = joblib.load(URG_LOGREG_MODEL)
urg_logreg_char = joblib.load(URG_LOGREG_CHAR)
urg_logreg_word = joblib.load(URG_LOGREG_WORD)

urg_rf_model = joblib.load(URG_RF_MODEL)
urg_rf_vec = joblib.load(URG_RF_VEC)


# ============================================================
# 1. DEPARTMENT — LOGISTIC REGRESSION
# ============================================================

def predict_department_logreg(text: str):
    char_features = dept_logreg_char.transform([text])
    word_features = dept_logreg_word.transform([text])

    X = hstack([char_features, word_features])

    prediction = dept_logreg_model.predict(X)[0]
    probability = dept_logreg_model.predict_proba(X)[0]

    return {
        "department": str(prediction),
        "confidence": round(float(probability.max()), 4)
    }


# ============================================================
# 2. DEPARTMENT — RANDOM FOREST
# ============================================================

def predict_department_rf(text: str):
    X = dept_rf_vec.transform([text])

    prediction = dept_rf_model.predict(X)[0]
    probability = dept_rf_model.predict_proba(X)[0]

    return {
        "department": str(prediction),
        "confidence": round(float(probability.max()), 4)
    }


# ============================================================
# 3. URGENCY — LOGISTIC REGRESSION
# ============================================================

def predict_urgency_logreg(text: str):
    char_features = urg_logreg_char.transform([text])
    word_features = urg_logreg_word.transform([text])

    X = hstack([char_features, word_features])

    prediction = urg_logreg_model.predict(X)[0]
    probability = urg_logreg_model.predict_proba(X)[0]

    return {
        "urgency": str(prediction),
        "confidence": round(float(probability.max()), 4)
    }


# ============================================================
# 4. URGENCY — RANDOM FOREST
# ============================================================

def predict_urgency_rf(text: str):
    X = urg_rf_vec.transform([text])

    prediction = urg_rf_model.predict(X)[0]
    probability = urg_rf_model.predict_proba(X)[0]

    return {
        "urgency": str(prediction),
        "confidence": round(float(probability.max()), 4)
    }