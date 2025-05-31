"""Main Samus agent implementation."""

import time
from dataclasses import dataclass
from typing import List

from .config import Config
from .core import MinimalAgentCore
from .mcp import MCPManager
from .models import ModelRouter


@dataclass
class AgentResponse:
    """Response from the Samus agent."""
    content: str
    mcps_used: List[str]
    execution_time: float
    reasoning_trace: List[str]


class SamusAgent:
    """Samus autonomous agent with dynamic MCP capabilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.config.ensure_directories()
        
        # Validate API keys
        api_status = config.validate_api_keys()
        if not api_status["openrouter"]:
            raise ValueError("OpenRouter API key is required")
        
        # Initialize core components
        self.model_router = ModelRouter(config)
        self.mcp_manager = MCPManager(config, self.model_router)
        self.core = MinimalAgentCore(config, self.model_router, self.mcp_manager)
    
    async def process(self, prompt: str) -> AgentResponse:
        """Process a prompt through the agent."""
        start_time = time.time()
        
        try:
            # Use the minimal agent core to solve the problem
            result = await self.core.solve_problem(prompt)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                content=result.content,
                mcps_used=result.mcps_used,
                execution_time=execution_time,
                reasoning_trace=result.reasoning_trace,
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResponse(
                content=f"Error processing request: {str(e)}",
                mcps_used=[],
                execution_time=execution_time,
                reasoning_trace=[f"Error: {str(e)}"],
            )
    
    async def shutdown(self) -> None:
        """Shutdown the agent and all its MCPs."""
        await self.mcp_manager.shutdown()