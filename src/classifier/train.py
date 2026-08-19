import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.classifier.features import extract_features

MODEL_DIR = "src/classifier/models"
MODEL_PATH = os.path.join(MODEL_DIR, "router_rf.joblib")

def train_model():
    print("Loading datasets...")
    
    # 1. Load both CSV files
    try:
        df_original = pd.read_csv("data/training_data.csv")
    except FileNotFoundError:
        df_original = pd.DataFrame(columns=["prompt", "tier"])
        
    try:
        df_gpt = pd.read_csv("data/gpt_training_data.csv")
    except FileNotFoundError:
        df_gpt = pd.DataFrame(columns=["prompt", "tier"])
    
    # 2. Combine and deduplicate
    df = pd.concat([df_original, df_gpt], ignore_index=True)
    initial_len = len(df)
    
    # Keep the last occurrence to ensure any corrected labels take precedence
    df = df.drop_duplicates(subset=["prompt"], keep="last").reset_index(drop=True)
    
    print(f"Combined dataset length: {len(df)} (Removed {initial_len - len(df)} duplicates)")
    print("\nClass distribution:")
    print(df["tier"].value_counts())
    
    # 3. Extract features
    print("\nGenerating features from prompts...")
    features_list = df["prompt"].apply(extract_features).tolist()
    X = pd.DataFrame(features_list)
    y = df["tier"]
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train Random Forest
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy on Test Set: {acc * 100:.1f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # 8. Overwrite the master dataset so future retrains have the combined data
    df.to_csv("data/training_data.csv", index=False)
    print("Master dataset saved to data/training_data.csv.")

if __name__ == "__main__":
    train_model()