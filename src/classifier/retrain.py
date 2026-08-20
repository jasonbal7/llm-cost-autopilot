import os
import pandas as pd
from src.classifier.train import train_model

TRAINING_DATA_PATH = "data/training_data.csv"
FEEDBACK_DATA_PATH = "data/feedback_failures.csv"

def retrain_from_failures():
    # If the feedback file doesn't exist or is empty, there is nothing to learn
    if not os.path.exists(FEEDBACK_DATA_PATH):
        print("No feedback failures found to retrain.")
        return

    failures_df = pd.read_csv(FEEDBACK_DATA_PATH)
    if failures_df.empty:
        print("Feedback failures log is empty.")
        return

    # Load the original dataset
    main_df = pd.read_csv(TRAINING_DATA_PATH)
    
    # Extract just the prompt and the NEW corrected tier from the failures
    new_examples = failures_df[["prompt", "tier"]].drop_duplicates()
    
    # Combine the old data with the newly corrected data
    combined_df = pd.concat([main_df, new_examples], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["prompt"], keep="last")
    
    # Save the updated, smarter dataset back to disk
    combined_df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"Appended {len(new_examples)} failure examples to {TRAINING_DATA_PATH}.")
    
    # Call the training function to rebuild the .joblib file with the corrected data
    train_model()
    
    # Erase the feedback buffer so we don't process the same failures twice next time
    with open(FEEDBACK_DATA_PATH, "w") as f:
        f.write("prompt,tier,source,logged_at\n")
    print("Feedback buffer cleared. Retraining complete.")

if __name__ == "__main__":
    retrain_from_failures()