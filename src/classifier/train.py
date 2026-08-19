import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.classifier.features import extract_features

def train_model():
    print("Loading dataset from data/training_data.csv...")
    if not os.path.exists("data/training_data.csv"):
        raise FileNotFoundError("Training data not found. Run generate_data.py first.")
        
    df = pd.read_csv("data/training_data.csv")
    
    print("Generating features from prompts...")
    # Convert prompts to feature dictionaries, then to a DataFrame
    features_df = pd.DataFrame([extract_features(p) for p in df['prompt']])
    X = features_df
    y = df['tier']
    
    # Split data (80% for training, 20% for testing the accuracy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Save the model
    os.makedirs("src/classifier/models", exist_ok=True)
    model_path = "src/classifier/models/router_rf.joblib"
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_model()