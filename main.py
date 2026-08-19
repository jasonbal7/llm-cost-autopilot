import asyncio
from src.core.router import PromptRouter

async def run_multi_test():
    print("Initializing PromptRouter...")
    router = PromptRouter()

    # Testing all three complexity tiers
    prompts = [
        # Expected Tier 1 (Low) - Simple extraction
        "Extract the email and phone number from this text: Call support at 555-0199 or email help@system.org.",
        
        # Expected Tier 2 (Medium) - Summarization/Constraints
        "Summarize the plot of the 1999 movie The Matrix in exactly three concise bullet points.",
        
        # Expected Tier 3 (High) - Deep Reasoning (We know this one works!)
        "Explain the exact architectural differences between a write-through and write-back cache."
    ]
    
    for i, prompt in enumerate(prompts, start=1):
        print(f"\n[Test {i}] Intercepting Prompt: '{prompt[:50]}...'")
        response = await router.route(prompt)
        
        print(f"Targeted Model : {response.model_id}")
        print(f"API Cost       : ${response.cost_usd:.6f}")
        print(f"Output Snippet : {response.content[:80]}...\n")
        print("-" * 60)
    
    print("[Waiting for background QualityVerifier to complete...]")
    pending_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending_tasks:
        await asyncio.gather(*pending_tasks)
    print("\nVerification complete! Check data/escalation_log.csv for updates.")

if __name__ == "__main__":
    asyncio.run(run_multi_test())