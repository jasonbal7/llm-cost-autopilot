import yaml
import asyncio
from src.classifier.predict import predict_complexity
from src.core.registry import MODEL_REGISTRY
from src.providers.unified import send_request, ModelResponse
from src.verifier.verifier import QualityVerifier
from src.classifier.features import extract_features

class PromptRouter:
    def __init__(self, config_path="configs/routing.yaml"):
        # Load the YAML configuration file so we know which models map to which tiers
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        # Initialize our background verifier, defaulting to GPT-4o as the judge
        self.verifier = QualityVerifier(
            judge_model_key=self.config.get("fallback", "gpt-4o"),
            threshold=0.80
        )

    async def route(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> ModelResponse:
        # 1. Ask the machine learning model which tier this prompt belongs to (low, medium, high)
        tier = predict_complexity(prompt)
        
        # --- NEW DEBUGGING PRINTS ---
        features = extract_features(prompt)
        print(f"\n[DEBUG] Extracted Features : {features}")
        print(f"[DEBUG] Model Predicted    : {tier.upper()}")
        # ----------------------------
        
        # 2. Look up the specific model ID assigned to that tier in the YAML file
        model_key = self.config["tiers"].get(tier, self.config["fallback"])
        
        # Safety check: if the YAML has a typo, use the fallback model
        if model_key not in MODEL_REGISTRY:
            model_key = self.config["fallback"]
            
        target_config = MODEL_REGISTRY[model_key]
        
        # 3. Send the API request to the chosen model and wait for the answer
        response = await send_request(prompt, target_config, system_prompt=system_prompt)
        
        # 4. Fire-and-forget: Start the background verification loop asynchronously.
        # This does not block the return statement below, meaning the user gets their answer instantly!
        asyncio.create_task(self.verifier.verify_async(prompt, response, tier))
        
        return response