# import os
# import joblib

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# def load_models():
#     """
#     Returns:
#       - category_char_vectorizer : char TF-IDF vectorizer (used by final category model)
#       - category_word_vectorizer : word TF-IDF vectorizer (used by final category model)
#       - category_model           : final LogisticRegression (trained on hstack([char, word]))
#       - category_rf_vectorizer   : word TF-IDF for the RF baseline
#       - category_rf_model        : RandomForest classifier
#       - urgency_rf_vectorizer    : word TF-IDF for urgency RF
#       - urgency_rf_model         : RandomForest urgency classifier
#     """
#     # ---- Final category model (combined char + word) ----
#     # UPDATED: File names changed to match the provided image
#     category_char_vectorizer = joblib.load(
#         os.path.join(MODELS_DIR, "category_final_char_vectorizer.joblib")
#     )
#     category_word_vectorizer = joblib.load(
#         os.path.join(MODELS_DIR, "category_final_word_vectorizer.joblib")
#     )
#     category_model = joblib.load(
#         os.path.join(MODELS_DIR, "category_final_model.joblib")
#     )

#     # ---- Category Random Forest (baseline / alternate) ----
#     category_rf_vectorizer = joblib.load(
#         os.path.join(MODELS_DIR, "category_rf_vectorizer.joblib")
#     )
#     category_rf_model = joblib.load(
#         os.path.join(MODELS_DIR, "category_rf_model.joblib")
#     )

#     # ---- Urgency Random Forest ----
#     urgency_rf_vectorizer = joblib.load(
#         os.path.join(MODELS_DIR, "urgency_rf_vectorizer.joblib")
#     )
#     urgency_rf_model = joblib.load(
#         os.path.join(MODELS_DIR, "urgency_rf_model.joblib")
#     )

#     return (
#         category_char_vectorizer,
#         category_word_vectorizer,
#         category_model,
#         category_rf_vectorizer,
#         category_rf_model,
#         urgency_rf_vectorizer,
#         urgency_rf_model,
#     )