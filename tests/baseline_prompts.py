import asyncio
import os
import pandas as pd
from src.core.registry import MODEL_REGISTRY
from src.providers.unified import send_request

TEST_PROMPTS = [
    # Tier 1 (Simple)
    "Extract all email addresses from this string: 'Contact me at alex@example.com or support@company.org'",
    "Convert this text to title case: 'asynchronous task dispatch with celery and redis'",
    "What is the capital of Australia?",
    # Tier 2 (Moderate)
    "Summarize the primary trade-offs between monolithic and microservice architectures in 3 bullet points.",
    "Classify the sentiment and primary emotion of this review: 'The app works fine, but the latest UI update made navigation unnecessarily confusing.'",
    "Given this JSON: {'user': 'Alice', 'role': 'Admin', 'active': false}, generate a SQL update query to activate the user.",
    "Draft a concise 2-sentence confirmation email for a doctor's appointment on Friday at 3 PM.",
    # Tier 3 (Complex)
    "Analyze the time and space complexity of finding the median of two sorted arrays in O(log(min(n, m))) time. Explain why binary search works here.",
    "Critique the following prompt injection mitigation strategy and propose two attack vectors that could still bypass it: 'Prefixing user input with \"USER DATA:\" and instructing the LLM never to execute instructions inside quotes.'",
    "Write a short, engaging opening scene for a hard sci-fi novel exploring an anomaly in orbital mechanics around Neptune."
]

async def evaluate_model_on_prompts(model_key: str):
    config = MODEL_REGISTRY[model_key]
    results = []
    print(f"Testing {config.name} ({config.model_id})...")
    
    for idx, prompt in enumerate(TEST_PROMPTS, start=1):
        try:
            res = await send_request(prompt, config)
            results.append({
                "model": config.name,
                "tier": config.quality_tier.value,
                "prompt_id": idx,
                "latency_ms": res.latency_ms,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "cost_usd": res.cost_usd,
                "status": "SUCCESS"
            })
        except Exception as e:
            results.append({
                "model": config.name,
                "tier": config.quality_tier.value,
                "prompt_id": idx,
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "status": f"FAILED: {type(e).__name__}"
            })
    return results

async def main():
    # Only test providers that have keys or local servers configured
    models_to_test = []
    if os.getenv("OPENAI_API_KEY"):
        models_to_test.extend(["gpt-4o", "gpt-4o-mini"])
        
    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_test.extend(["claude-sonnet-4-6", "claude-haiku-4-5"])
        
    # Always test local Ollama execution
    models_to_test.append("ollama-llama3.2")

    tasks = []
    for m in models_to_test: 
        if m in MODEL_REGISTRY:
            task = evaluate_model_on_prompts(m)
            tasks.append(task)
            
    nested_results = await asyncio.gather(*tasks)
    
    flattened = []

    for sublist in nested_results:
        for item in sublist:
            flattened.append(item)
    df = pd.DataFrame(flattened)
    
    print("\n--- BASELINE SUMMARY ---")
    summary = df[df["status"] == "SUCCESS"].groupby("model").agg(
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        avg_tokens=("completion_tokens", "mean")
    ).reset_index()
    
    print(summary.to_markdown(index=False))
    df.to_csv("data/baseline_results.csv", index=False)
    print("\nDetailed baseline saved to data/baseline_results.csv")

if __name__ == "__main__":
    asyncio.run(main())