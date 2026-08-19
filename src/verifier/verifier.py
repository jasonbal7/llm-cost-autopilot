import os
import csv
import asyncio
import pandas as pd
from datetime import datetime, timezone  # Updated to fix the deprecation warning
from src.core.registry import MODEL_REGISTRY
from src.providers.unified import send_request, ModelResponse
from src.verifier.evaluators import evaluate_response

# File paths for our tracking datasets
ESCALATION_LOG_PATH = "data/escalation_log.csv"
FAILURE_DATASET_PATH = "data/feedback_failures.csv"

def _ensure_csv_headers():
    """A helper function that creates the CSV files and writes the header rows if they don't exist yet."""
    os.makedirs("data", exist_ok=True)
    
    # Initialize the audit log for all background checks
    if not os.path.exists(ESCALATION_LOG_PATH):
        with open(ESCALATION_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "prompt", "original_model", "judge_model",
                "score", "passed", "cost_delta_usd", "reason"
            ])
            
    # Initialize the specific dataset that will be used for retraining
    if not os.path.exists(FAILURE_DATASET_PATH):
        with open(FAILURE_DATASET_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "tier", "source", "logged_at"])

class QualityVerifier:
    def __init__(self, judge_model_key: str = "gpt-4o", threshold: float = 0.80):
        self.judge_model_key = judge_model_key
        self.threshold = threshold
        _ensure_csv_headers()

    async def verify_async(self, prompt: str, initial_response: ModelResponse, predicted_tier: str):
        """Asynchronously runs verification against the frontier model in the background."""
        
        # If the ML model already predicted "high" and routed to the frontier model initially,
        # there is no need to verify it against itself. We just exit the function early.
        if initial_response.model_id == MODEL_REGISTRY[self.judge_model_key].model_id:
            return

        judge_config = MODEL_REGISTRY[self.judge_model_key]
        
        # Silently ask the expensive model the same exact question the user asked
        reference_response = await send_request(prompt, judge_config)
        
        # Compare the cheap answer against the expensive answer
        eval_result = await evaluate_response(
            prompt=prompt,
            candidate_response=initial_response.content,
            reference_response=reference_response.content,
            judge_model_key=self.judge_model_key,
            threshold=self.threshold
        )
        
        # Calculate exactly how much money we saved by using the cheap model first
        cost_delta = reference_response.cost_usd - initial_response.cost_usd
        
        # Log every single verification event (pass or fail) into our audit trail
        with open(ESCALATION_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(), # Fixed timezone awareness
                prompt,
                initial_response.model_id,
                judge_config.model_id,
                eval_result.score,
                eval_result.passed,
                round(cost_delta, 6),
                eval_result.reason
            ])

        # If the cheap model performed poorly, we need to log it so the system learns from the mistake
        if not eval_result.passed:
            # THIS IS THE SECRET SAUCE: We forcibly escalate the label.
            # If the ML model guessed "medium" and failed, the correct answer must be "high".
            escalated_tier = "high" if predicted_tier == "medium" else "medium"
            
            with open(FAILURE_DATASET_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    prompt,
                    escalated_tier,  # We log the *correct* escalated tier, not the original wrong one
                    "verifier_feedback",
                    datetime.now(timezone.utc).isoformat()
                ])
            print(f"\n[VERIFIER WARNING] Routing failure detected for prompt: '{prompt[:40]}...'")
            print(f"Score: {eval_result.score:.2f} | Reason: {eval_result.reason}")
            print(f"Logged to {FAILURE_DATASET_PATH} for model retraining.")