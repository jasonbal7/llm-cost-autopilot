import json
from dataclasses import dataclass
from typing import Optional
from src.core.registry import MODEL_REGISTRY
from src.providers.unified import send_request

# We use a dataclass to strictly define what an evaluation result looks like.
@dataclass
class EvalResult:
    score: float                # A normalized score between 0.0 (terrible) and 1.0 (perfect)
    passed: bool                # A simple boolean flag (True if score is >= our threshold)
    reason: str                 # The LLM's explanation for why it gave that score
    judge_model: str            # Which model performed the evaluation (e.g., GPT-4o)
    task_type: str              # The category of the prompt (extraction, summary, etc.)

# This is the strict system instruction that forces the frontier model to act as an objective judge.
JUDGE_SYSTEM_PROMPT = """You are an expert AI evaluator comparing two model outputs.
- Response A: The candidate answer to evaluate.
- Response B: The gold-standard reference answer.

Determine the task category (e.g., 'extraction', 'summarization', 'classification', 'reasoning', 'creative', 'general').
Evaluate Response A against Response B based on accuracy, completeness, and formatting constraint adherence.

Output ONLY valid JSON with this exact schema:
{
  "score": <float between 0.0 and 1.0>,
  "passed": <true/false based on whether score >= 0.80>,
  "task_type": "<determined category>",
  "reason": "<brief explanation of the score>"
}
"""

async def evaluate_response(prompt: str,candidate_response: str, reference_response: str, judge_model_key: str = "gpt-4o", threshold: float = 0.80
) -> EvalResult:
    """Evaluates the cheap model's response against the expensive model's response."""
    # Look up the judge model's configuration (pricing, provider) from our registry
    judge_config = MODEL_REGISTRY[judge_model_key]
    
    # Construct the user message injecting both answers so the judge can compare them
    eval_user_prompt = f"""User Prompt:
        \"\"\"{prompt}\"\"\"

        Response A (Candidate):
        \"\"\"{candidate_response}\"\"\"

        Response B (Reference Standard):
        \"\"\"{reference_response}\"\"\"
        """
    
    # Send the evaluation request to the judge model
    judge_res = await send_request(eval_user_prompt, judge_config, system_prompt=JUDGE_SYSTEM_PROMPT)
    
    try:
        # LLMs sometimes wrap JSON in markdown blocks (```json ... ```). 
        # This string manipulation strips those out so the Python json library doesn't crash.
        raw_json = judge_res.content.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json.replace("```json", "").replace("```", "").strip()
        elif raw_json.startswith("```"):
            raw_json = raw_json.replace("```", "").strip()
            
        # Convert the string into a Python dictionary
        data = json.loads(raw_json)
        score = float(data.get("score", 0.0))
        
        # Return the structured dataclass back to the verifier
        return EvalResult(
            score=score,
            passed=score >= threshold,
            reason=data.get("reason", "No reason provided"),
            judge_model=judge_config.name,
            task_type=data.get("task_type", "general")
        )
    except Exception as e:
        # If the LLM failed to return valid JSON, we treat it as an automatic failure for safety
        return EvalResult(
            score=0.0,
            passed=False,
            reason=f"Failed to parse judge output: {str(e)}",
            judge_model=judge_config.name,
            task_type="unknown"
        )