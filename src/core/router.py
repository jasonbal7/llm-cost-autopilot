import yaml
import csv
import os
import hashlib
from datetime import datetime, timezone
from src.classifier.predict import predict_complexity
from src.core.registry import MODEL_REGISTRY
from src.providers.unified import send_request, ModelResponse
from src.verifier.evaluators import evaluate_response

MASTER_LOG_PATH = "data/master_audit_log.csv"
ESCALATION_LOG_PATH = "data/escalation_log.csv"
FEEDBACK_LOG_PATH = "data/feedback_failures.csv"

def _ensure_logs():
    """Ensure all logs exist with their updated headers, even if the file is blank."""
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(MASTER_LOG_PATH) or os.path.getsize(MASTER_LOG_PATH) == 0:
        with open(MASTER_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "tier_predicted", "model_used", 
                "actual_cost", "hypothetical_max_cost", "savings_usd", "latency_ms", "escalated"
            ])
            
    if not os.path.exists(ESCALATION_LOG_PATH) or os.path.getsize(ESCALATION_LOG_PATH) == 0:
        with open(ESCALATION_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "prompt_hash", "task_type", "original_model", "judge_model",
                "score", "passed", "cost_delta_usd", "reason"
            ])
            
    if not os.path.exists(FEEDBACK_LOG_PATH) or os.path.getsize(FEEDBACK_LOG_PATH) == 0:
        with open(FEEDBACK_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["prompt", "tier", "source", "logged_at"])

class PromptRouter:
    def __init__(self, config_path="configs/routing.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.judge_key = self.config.get("fallback", "gpt-4o")
        self.threshold = 0.80
        _ensure_logs()

    async def route(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> ModelResponse:
        tier = predict_complexity(prompt)
        model_key = self.config["tiers"].get(tier, self.config["fallback"])
        
        if model_key not in MODEL_REGISTRY:
            model_key = self.config["fallback"]
            
        target_config = MODEL_REGISTRY[model_key]
        judge_config = MODEL_REGISTRY[self.judge_key]
        
        # Phase 4: Hash the prompt for enterprise privacy
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        
        # 1. Fast generation
        candidate = await send_request(prompt, target_config, system_prompt)
        
        # 2. Prepare default logging metrics
        final_response = candidate
        escalated = False
        
        # Fix: Calculate savings vs the absolute most expensive premium model (Sonnet 3.5)
        max_tier_config = MODEL_REGISTRY["claude-sonnet-4-6"]
        hypothetical_cost = (candidate.prompt_tokens * max_tier_config.cost_per_input_token) + \
                            (candidate.completion_tokens * max_tier_config.cost_per_output_token)
        savings = hypothetical_cost - candidate.cost_usd

        # 3. Blocking Verification
        if target_config.model_id != judge_config.model_id:
            reference = await send_request(prompt, judge_config, system_prompt)
            
            eval_res = await evaluate_response(
                prompt=prompt, 
                candidate_response=candidate.content, 
                reference_response=reference.content, 
                judge_model_key=self.judge_key, 
                threshold=self.threshold
            )   
            
            cost_delta = reference.cost_usd - candidate.cost_usd
            
            # Log the background check (using prompt_hash)
            with open(ESCALATION_LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now(timezone.utc).isoformat(), prompt_hash, eval_res.task_type, 
                    candidate.model_id, judge_config.model_id, eval_res.score, 
                    eval_res.passed, round(cost_delta, 6), eval_res.reason
                ])

            # 4. Handle Failure: Escalate and Replace!
            if not eval_res.passed:
                escalated = True
                final_response = reference  
                savings = 0.0               
                
                # Fix: Proper escalation logic to prevent downgrading High tier
                if tier == "low":
                    escalated_tier = "medium"
                else:
                    escalated_tier = "high"
                    
                with open(FEEDBACK_LOG_PATH, "a", newline="") as f:
                    # Note: We keep the raw prompt here so the ML model can actually train on it later!
                    csv.writer(f).writerow([prompt, escalated_tier, "auto_escalation", datetime.now(timezone.utc).isoformat()])
                
                print(f"\n[ESCALATION TRIGGERED] {candidate.model_id} failed {eval_res.task_type} check. Score: {eval_res.score}.")
                print(f"Reason: {eval_res.reason}")
                print(f"Returning superior {self.judge_key} result to user.")
        
        # 5. Master Audit Log
        with open(MASTER_LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(), tier, final_response.model_id, 
                round(final_response.cost_usd, 6), round(hypothetical_cost, 6), 
                round(savings, 6), final_response.latency_ms, escalated
            ])
            
        return final_response