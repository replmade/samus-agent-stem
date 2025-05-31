"""Configuration management for Samus agent."""

import os
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration for Samus agent."""
    
    # API Keys
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # Model Configuration (OpenRouter model names)
    supervisor_model: str = "anthropic/claude-sonnet-4"
    default_lightweight_model: str = "anthropic/claude-3.5-haiku"
    default_moderate_model: str = "anthropic/claude-sonnet-4"
    default_expert_model: str = "anthropic/claude-opus-4"
    
    # MCP Configuration
    mcp_repository_path: str = Field(default_factory=lambda: str(Path.home() / ".samus" / "mcps"))
    data_directory: str = Field(default_factory=lambda: str(Path.home() / ".samus" / "data"))
    max_mcps_in_memory: int = 50
    mcp_cache_ttl: int = 3600  # seconds
    
    # Performance Settings
    max_concurrent_mcps: int = 5
    request_timeout: int = 30
    max_retries: int = 3
    
    # Capability Discovery
    github_token: Optional[str] = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))
    capability_search_limit: int = 10
    
    # Logging
    log_level: str = "INFO"
    enable_performance_tracking: bool = True
    
    # MCP Generation Control
    enable_mcp_generation: bool = True
    mcp_generation_timeout: float = 60.0
    mcp_execution_timeout: float = 30.0
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from file or environment."""
        if config_path and Path(config_path).exists():
            # TODO: Implement JSON/YAML config file loading
            pass
        
        # For now, use environment variables and defaults
        return cls()
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present."""
        return {
            "anthropic": bool(self.anthropic_api_key),
            "openai": bool(self.openai_api_key),
            "openrouter": bool(self.openrouter_api_key),
        }
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        Path(self.mcp_repository_path).mkdir(parents=True, exist_ok=True)
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)