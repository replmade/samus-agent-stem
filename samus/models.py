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
        
        # Model catalog with latest models (as of May 2025)
        self.model_catalog = {
            # Anthropic models
            "anthropic": {
                "claude-3-5-haiku": {
                    "cost": "low", 
                    "speed": "fastest", 
                    "reasoning": "advanced", 
                    "context": "200k"
                },
                "claude-sonnet-4": {
                    "cost": "medium", 
                    "speed": "fast", 
                    "reasoning": "expert", 
                    "context": "200k", 
                    "max_output": "64k"
                },
                "claude-opus-4": {
                    "cost": "high", 
                    "speed": "medium", 
                    "reasoning": "expert", 
                    "context": "200k", 
                    "max_output": "32k"
                }
            },
            # OpenAI models (future integration)
            "openai": {
                "o4-mini": {
                    "cost": "low", 
                    "speed": "fastest", 
                    "reasoning": "expert", 
                    "context": "128k", 
                    "thinking": True
                },
                "gpt-4.5": {
                    "cost": "medium", 
                    "speed": "fast", 
                    "reasoning": "expert", 
                    "context": "1M"
                },
                "o3": {
                    "cost": "high", 
                    "speed": "slow", 
                    "reasoning": "expert", 
                    "context": "128k", 
                    "thinking": True
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
        # For now, default to Claude 3.5 Haiku for lightweight tasks
        return "claude-3-5-haiku"
    
    def _select_moderate_model(self, capabilities: List[str]) -> str:
        """Balance cost and capability for moderate complexity."""
        return "claude-sonnet-4"
    
    def _select_expert_model(self, capabilities: List[str]) -> str:
        """Use most capable models for complex reasoning."""
        return "claude-opus-4"
    
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
            return self._call_openrouter_model(model, messages, system, max_tokens, temperature)
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