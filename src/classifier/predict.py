import os
import pandas as pd
import joblib
from src.classifier.features import extract_features

# Load model into memory once when the app starts
MODEL_PATH = "src/classifier/models/router_rf.joblib"

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
        _classifier = joblib.load(MODEL_PATH)
    return _classifier

def predict_complexity(prompt: str) -> str:
    """Predicts 'low', 'medium', or 'high' complexity for a given prompt."""
    clf = get_classifier()
    features = extract_features(prompt)
    
    # Scikit-learn expects a 2D array/DataFrame
    X_new = pd.DataFrame([features])
    
    prediction = clf.predict(X_new)[0]
    return prediction           