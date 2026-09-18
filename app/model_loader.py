# app/model_loader.py
import os
# pyrefly: ignore [missing-import]
import joblib

# Resolve <project_root>/models/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_models():
    """
    Load the 4 joblib artifacts:
      - category: 1 full pipeline (raw text → label)
      - urgency:  1 vectorizer + 1 model (raw text → vectorize → label)
    """
    category_model = joblib.load(
        os.path.join(MODELS_DIR, "category_char_model.joblib")
    )
    urgency_vectorizer = joblib.load(
        os.path.join(MODELS_DIR, "urgency_rf_vectorizer.joblib")
    )
    urgency_model = joblib.load(
        os.path.join(MODELS_DIR, "urgency_rf_model.joblib")
    )
    return category_model, urgency_vectorizer, urgency_model