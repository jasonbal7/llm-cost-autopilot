from dataclasses import dataclass
from enum import Enum
from typing import Dict

class QualityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    
class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    
@dataclass(frozen=True)             # makes the object immutable after it's being created
class ModelConfig:
    name: str
    model_id: str
    provider: ProviderType
    cost_per_input_token: float     # USD per single token
    cost_per_output_token: float    # USD per single ouput token
    quality_tier: QualityTier
    expected_latency_ms: float = 0.0
    
# Base Registry (Rates per 1M tokens converted to per-token USD)
MODEL_REGISTRY = {
    "claude-sonnet-4-6": ModelConfig(
        name="Claude Sonnet 4.6",
        model_id="claude-sonnet-4-6",
        provider=ProviderType.ANTHROPIC,
        cost_per_input_token=3.00 / 1_000_000,
        cost_per_output_token=15.00 / 1_000_000,
        quality_tier=QualityTier.HIGH,
        expected_latency_ms=1100.0,
    ),
    "claude-haiku-4-5": ModelConfig(
        name="Claude Haiku 4.5",
        model_id="claude-haiku-4-5-20251001",
        provider=ProviderType.ANTHROPIC,
        cost_per_input_token=1.00 / 1_000_000,
        cost_per_output_token=5.00 / 1_000_000,
        quality_tier=QualityTier.LOW,
        expected_latency_ms=400.0,
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        model_id="gpt-4o",
        provider=ProviderType.OPENAI,
        cost_per_input_token=2.50 / 1_000_000,
        cost_per_output_token=10.00 / 1_000_000,
        quality_tier=QualityTier.HIGH,
        expected_latency_ms=800.0,
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        model_id="gpt-4o-mini",
        provider=ProviderType.OPENAI,
        cost_per_input_token=0.15 / 1_000_000,
        cost_per_output_token=0.60 / 1_000_000,
        quality_tier=QualityTier.MEDIUM,
        expected_latency_ms=300.0,
    ),
    "ollama-llama3.2": ModelConfig(
        name="Llama 3.2 3B (Local)",
        model_id="llama3.2:3b",
        provider=ProviderType.LOCAL,
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        quality_tier=QualityTier.LOW,
        expected_latency_ms=2000.0,
    )
}