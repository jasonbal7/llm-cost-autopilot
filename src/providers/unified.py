import os
import time
from dataclasses import dataclass
from typing import Optional
import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from src.core.registry import ModelConfig, ProviderType
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelResponse:
    
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float 
    model_id: str
    provider: str
    raw_response: Optional[dict] = None
    

# Reusable clients
_openai_client: Optional[AsyncOpenAI] = None
_anthropic_client: Optional[AsyncAnthropic] = None

def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _openai_client

def get_anthropic_client() -> AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    return _anthropic_client


async def send_request(prompt: str, config: ModelConfig, system_prompt: str = "You are a helpful assistant.") -> ModelResponse:
    """Dispatches a prompt to the specified provider and returns a normalized response."""
    start_time = time.perf_counter()
    
    if config.provider == ProviderType.OPENAI:
        client = get_openai_client()                                                                
        res = await client.chat.completions.create(
            model=config.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        prompt_tokens = res.usage.prompt_tokens
        completion_tokens = res.usage.completion_tokens
        content = res.choices[0].message.content or ""

    elif config.provider == ProviderType.ANTHROPIC:
        client = get_anthropic_client()
        res = await client.messages.create(
            model=config.model_id,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.2,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        prompt_tokens = res.usage.input_tokens
        completion_tokens = res.usage.output_tokens
        content = "".join([block.text for block in res.content if hasattr(block, "text")])

    elif config.provider == ProviderType.LOCAL:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
        payload = {
            "model": config.model_id,
            "prompt": f"{system_prompt}\n\nUser: {prompt}\nAssistant:",
            "stream": False,
            "options": {"temperature": 0.2}
        }
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            resp = await http_client.post(ollama_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        content = data.get("response", "")

    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

    # Cost Calculation
    cost = (prompt_tokens * config.cost_per_input_token) + (completion_tokens * config.cost_per_output_token)

    return ModelResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost,
        latency_ms=round(latency_ms, 2),
        model_id=config.model_id,
        provider=config.provider.value,
    )