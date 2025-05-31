"""Model routing and management."""

from typing import Dict, List, Optional

import anthropic
import openai
from openai import OpenAI

from .config import Config


class ModelRouter:
    """
    Manages multiple model providers and routes requests to optimal models.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize clients - use OpenRouter for all models
        self.openrouter_client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        ) if config.openrouter_api_key else None
        
        # Fallback to direct clients if needed
        self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key) if config.anthropic_api_key else None
        self.openai_client = OpenAI(api_key=config.openai_api_key) if config.openai_api_key else None
        
        # Model catalog with latest models from OpenRouter (updated May 2025)
        self.model_catalog = {
            # Anthropic models - Current models available on OpenRouter
            "anthropic": {
                "claude-3.5-haiku": {
                    "openrouter_id": "anthropic/claude-3.5-haiku",
                    "cost": "very_low",  # $0.0008 per 1M prompt tokens
                    "speed": "fastest", 
                    "reasoning": "advanced", 
                    "context": "200k",
                    "max_output": "8k",
                    "multimodal": True,
                    "pricing": {"prompt": 0.0000008, "completion": 0.000004}
                },
                "claude-3.5-sonnet": {
                    "openrouter_id": "anthropic/claude-3.5-sonnet",
                    "cost": "low",  # $3 per 1M prompt tokens
                    "speed": "fast", 
                    "reasoning": "expert", 
                    "context": "200k", 
                    "max_output": "8k",
                    "multimodal": True,
                    "pricing": {"prompt": 0.000003, "completion": 0.000015}
                },
                "claude-3.7-sonnet": {
                    "openrouter_id": "anthropic/claude-3.7-sonnet",
                    "cost": "low",  # Same as 3.5 Sonnet
                    "speed": "fast", 
                    "reasoning": "expert", 
                    "context": "200k", 
                    "max_output": "64k",
                    "multimodal": True,
                    "thinking": True,  # Has thinking mode
                    "pricing": {"prompt": 0.000003, "completion": 0.000015}
                },
                "claude-sonnet-4": {
                    "openrouter_id": "anthropic/claude-sonnet-4",
                    "cost": "low",  # Same pricing as 3.5/3.7
                    "speed": "fast", 
                    "reasoning": "expert", 
                    "context": "200k", 
                    "max_output": "64k",
                    "multimodal": True,
                    "pricing": {"prompt": 0.000003, "completion": 0.000015}
                },
                "claude-opus-4": {
                    "openrouter_id": "anthropic/claude-opus-4",
                    "cost": "high",  # $15 per 1M prompt tokens
                    "speed": "medium", 
                    "reasoning": "expert", 
                    "context": "200k", 
                    "max_output": "32k",
                    "multimodal": True,
                    "pricing": {"prompt": 0.000015, "completion": 0.000075}
                },
                "claude-3-opus": {
                    "openrouter_id": "anthropic/claude-3-opus",
                    "cost": "high",  # Same as Opus 4
                    "speed": "medium", 
                    "reasoning": "expert", 
                    "context": "200k", 
                    "max_output": "4k",
                    "multimodal": True,
                    "pricing": {"prompt": 0.000015, "completion": 0.000075}
                }
            },
            # OpenAI models - Current models available on OpenRouter
            "openai": {
                "gpt-4o-mini": {
                    "openrouter_id": "openai/gpt-4o-mini",
                    "cost": "very_low",
                    "speed": "fastest", 
                    "reasoning": "advanced", 
                    "context": "128k",
                    "multimodal": True,
                    "pricing": {"prompt": 0.00000015, "completion": 0.0000006}
                },
                "gpt-4o": {
                    "openrouter_id": "openai/gpt-4o",
                    "cost": "medium",
                    "speed": "fast", 
                    "reasoning": "expert", 
                    "context": "128k",
                    "multimodal": True,
                    "pricing": {"prompt": 0.0000025, "completion": 0.00001}
                },
                "o1-mini": {
                    "openrouter_id": "openai/o1-mini",
                    "cost": "medium",
                    "speed": "slow", 
                    "reasoning": "expert", 
                    "context": "128k", 
                    "thinking": True,
                    "pricing": {"prompt": 0.000003, "completion": 0.000012}
                },
                "o1": {
                    "openrouter_id": "openai/o1",
                    "cost": "high",
                    "speed": "slow", 
                    "reasoning": "expert", 
                    "context": "200k", 
                    "thinking": True,
                    "pricing": {"prompt": 0.000015, "completion": 0.00006}
                },
                "o1-pro": {
                    "openrouter_id": "openai/o1-pro",
                    "cost": "very_high",
                    "speed": "slow", 
                    "reasoning": "expert", 
                    "context": "128k", 
                    "thinking": True,
                    "pricing": {"prompt": 0.00006, "completion": 0.00024}
                },
                "o3-mini": {
                    "openrouter_id": "openai/o3-mini",
                    "cost": "high",
                    "speed": "slow", 
                    "reasoning": "expert", 
                    "context": "128k", 
                    "thinking": True,
                    "pricing": {"prompt": 0.000013, "completion": 0.000052}
                }
            },
            # Google models - Current models available on OpenRouter
            "google": {
                "gemini-2.0-flash": {
                    "openrouter_id": "google/gemini-2.0-flash",
                    "cost": "very_low",
                    "speed": "fastest", 
                    "reasoning": "advanced", 
                    "context": "1M",
                    "multimodal": True,
                    "pricing": {"prompt": 0.000000075, "completion": 0.0000003}
                },
                "gemini-2.5-flash": {
                    "openrouter_id": "google/gemini-2.5-flash-preview",
                    "cost": "very_low",
                    "speed": "fastest", 
                    "reasoning": "expert", 
                    "context": "1M",
                    "multimodal": True,
                    "thinking": True,
                    "pricing": {"prompt": 0.00000015, "completion": 0.0000006}
                },
                "gemini-2.5-pro": {
                    "openrouter_id": "google/gemini-2.5-pro-preview",
                    "cost": "low",
                    "speed": "fast", 
                    "reasoning": "expert", 
                    "context": "1M",
                    "multimodal": True,
                    "pricing": {"prompt": 0.00000125, "completion": 0.00001}
                },
                "gemma-3-27b": {
                    "openrouter_id": "google/gemma-3-27b-it",
                    "cost": "very_low",
                    "speed": "fast", 
                    "reasoning": "advanced", 
                    "context": "128k",
                    "multimodal": True,
                    "pricing": {"prompt": 0.0000001, "completion": 0.0000002}
                }
            }
        }
    
    def select_optimal_model(
        self, 
        capabilities: List[str], 
        complexity: str, 
        requirements: Optional[Dict] = None
    ) -> str:
        """Select the best model based on task requirements and cost optimization."""
        
        if complexity == "lightweight":
            return self._select_lightweight_model(capabilities)
        elif complexity == "moderate":
            return self._select_moderate_model(capabilities)
        else:
            return self._select_expert_model(capabilities)
    
    def _select_lightweight_model(self, capabilities: List[str]) -> str:
        """Select fast, cost-effective models for simple tasks."""
        # Prioritize by cost and speed for lightweight tasks
        if "multimodal" in capabilities:
            return "anthropic/claude-3.5-haiku"  # Best multimodal lightweight option
        elif "thinking" in capabilities:
            return "google/gemini-2.5-flash-preview"  # Fastest thinking model
        else:
            return "google/gemini-2.0-flash"  # Cheapest and fastest for text-only
    
    def _select_moderate_model(self, capabilities: List[str]) -> str:
        """Balance cost and capability for moderate complexity."""
        if "thinking" in capabilities:
            return "anthropic/claude-3.7-sonnet"  # Has thinking mode
        elif "multimodal" in capabilities:
            return "anthropic/claude-3.5-sonnet"  # Excellent multimodal capabilities
        elif "large_context" in capabilities:
            return "google/gemini-2.5-pro-preview"  # 1M context window
        else:
            return "anthropic/claude-sonnet-4"  # Latest general-purpose model
    
    def _select_expert_model(self, capabilities: List[str]) -> str:
        """Use most capable models for complex reasoning."""
        if "thinking" in capabilities:
            return "openai/o1"  # Best reasoning with thinking
        elif "coding" in capabilities:
            return "anthropic/claude-opus-4"  # Best coding model
        elif "large_context" in capabilities:
            return "google/gemini-2.5-pro-preview"  # 1M context for complex analysis
        else:
            return "anthropic/claude-sonnet-4"  # Most balanced expert model
    
    def get_openrouter_model_id(self, model_name: str) -> str:
        """Get the OpenRouter model ID for a given model name."""
        # If it's already an OpenRouter ID, return as-is
        if "/" in model_name:
            return model_name
        
        # Search through our catalog to find the OpenRouter ID
        for provider, models in self.model_catalog.items():
            for short_name, model_info in models.items():
                if short_name == model_name:
                    return model_info.get("openrouter_id", model_name)
        
        # Fallback to the original name if not found
        return model_name
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get comprehensive information about a model."""
        # Search through our catalog to find the model info
        for provider, models in self.model_catalog.items():
            for short_name, model_info in models.items():
                if short_name == model_name or model_info.get("openrouter_id") == model_name:
                    return {
                        "name": short_name,
                        "provider": provider,
                        "openrouter_id": model_info.get("openrouter_id", model_name),
                        **model_info
                    }
        
        # Return minimal info if not found
        return {
            "name": model_name,
            "provider": "unknown",
            "openrouter_id": model_name,
            "cost": "unknown",
            "speed": "unknown",
            "reasoning": "unknown"
        }
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models organized by provider."""
        return {
            provider: list(models.keys())
            for provider, models in self.model_catalog.items()
        }
    
    def get_models_by_criteria(self, cost: str = None, speed: str = None, reasoning: str = None) -> List[str]:
        """Get models that match specific criteria."""
        matching_models = []
        
        for provider, models in self.model_catalog.items():
            for short_name, model_info in models.items():
                match = True
                
                if cost and model_info.get("cost") != cost:
                    match = False
                if speed and model_info.get("speed") != speed:
                    match = False
                if reasoning and model_info.get("reasoning") != reasoning:
                    match = False
                
                if match:
                    matching_models.append(short_name)
        
        return matching_models
    
    def call_model(
        self, 
        model: str, 
        messages: List[Dict], 
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> str:
        """Call the specified model with the given parameters."""
        
        # Use OpenRouter for all models if available
        if self.openrouter_client:
            openrouter_model = self.get_openrouter_model_id(model)
            return self._call_openrouter_model(openrouter_model, messages, system, max_tokens, temperature)
        elif model.startswith("claude") and self.anthropic_client:
            return self._call_anthropic_model(model, messages, system, max_tokens, temperature)
        elif (model.startswith("gpt") or model.startswith("o")) and self.openai_client:
            return self._call_openai_model(model, messages, system, max_tokens, temperature)
        else:
            raise ValueError(f"No available client for model: {model}")
    
    def _call_anthropic_model(
        self, 
        model: str, 
        messages: List[Dict], 
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Anthropic model."""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages
            )
            return response.content[0].text if response.content else ""
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
    
    def _call_openrouter_model(
        self, 
        model: str, 
        messages: List[Dict], 
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call model through OpenRouter."""
        # Add system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {str(e)}")
    
    def _call_openai_model(
        self, 
        model: str, 
        messages: List[Dict], 
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call OpenAI model."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized - missing API key")
        
        # Add system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by provider."""
        available = {}
        
        if self.anthropic_client:
            available["anthropic"] = list(self.model_catalog["anthropic"].keys())
        
        if self.openai_client:
            available["openai"] = list(self.model_catalog["openai"].keys())
        
        return available