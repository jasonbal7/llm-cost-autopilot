import asyncio
import pandas as pd
from src.providers.unified import get_openai_client

async def generate_prompts_for_tier(tier_name: str, description: str, count: int) -> list:
    """Uses GPT-4o-mini to generate synthetic user prompts for training."""
    client = get_openai_client()
    
    system_prompt = (
        "You are a synthetic data generator for an AI routing system. "
        "Generate a JSON list of user prompts."
    )
    
    user_prompt = (
        f"Generate exactly {count} diverse user prompts that fit this complexity tier: '{tier_name}'.\n"
        f"Tier description: {description}\n\n"
        "Output ONLY valid JSON in this format: [\"prompt 1\", \"prompt 2\", ...]"
    )
    
    print(f"Generating {count} prompts for {tier_name} tier...")
    
    # We use gpt-4o-mini because it's fast, cheap, and great at following JSON structures
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
    )
    
    # Parse the JSON string back into a Python list
    import json
    raw_content = response.choices[0].message.content.strip()
    
    # Strip markdown code blocks if the LLM added them
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:-3]
        
    prompts = json.loads(raw_content)
    return [{"prompt": p, "tier": tier_name} for p in prompts]

async def main():
    tiers = {
        "low": "Tier 1 (simple): reformatting, extraction, basic Q&A from provided context.",
        "medium": "Tier 2 (moderate): summarization, classification, structured analysis.",
        "high": "Tier 3 (complex): multi-step reasoning, creative generation, nuanced judgment calls."
    }
    
    all_data = []
    
    # Generate ~75 prompts per tier to get over the 200 target
    for tier_name, description in tiers.items():
        data = await generate_prompts_for_tier(tier_name, description, 75)
        all_data.extend(data)
        
    df = pd.DataFrame(all_data)
    
    # Save the dataset
    df.to_csv("data/gpt_training_data.csv", index=False)
    print(f"\nSuccessfully generated {len(df)} prompts and saved to data/gpt_training_data.csv")

if __name__ == "__main__":
    asyncio.run(main())