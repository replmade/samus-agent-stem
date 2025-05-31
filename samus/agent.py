"""Main Samus agent implementation."""

import time
from dataclasses import dataclass
from typing import List

from .config import Config
from .core import MinimalAgentCore
from .mcp import MCPManager, MCPSpecification
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
    
    def get_all_mcps(self) -> List[MCPSpecification]:
        """Get all MCPs from this agent for distillation purposes."""
        return self.mcp_manager.repository.get_all_mcps()
    
    def get_high_performance_mcps(self, performance_threshold: float = 0.7) -> List[MCPSpecification]:
        """Get MCPs that meet the performance threshold for distillation."""
        all_mcps = self.get_all_mcps()
        return [
            mcp for mcp in all_mcps 
            if mcp.performance_metrics.get("success_rate", 0.0) >= performance_threshold
        ]
    
    def get_agent_statistics(self) -> dict:
        """Get comprehensive statistics about this agent's capabilities."""
        all_mcps = self.get_all_mcps()
        
        if not all_mcps:
            return {
                "total_mcps": 0,
                "avg_success_rate": 0.0,
                "total_executions": 0,
                "domains": {}
            }
        
        total_executions = sum(mcp.performance_metrics.get("execution_count", 0) for mcp in all_mcps)
        success_rates = [mcp.performance_metrics.get("success_rate", 0.0) for mcp in all_mcps if mcp.performance_metrics.get("success_rate", 0.0) > 0]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        # Categorize by domain
        domains = {}
        for mcp in all_mcps:
            # Simple domain classification
            desc = mcp.description.lower()
            if "data" in desc or "analysis" in desc:
                domain = "data_analysis"
            elif "http" in desc or "api" in desc:
                domain = "web_apis"
            elif "math" in desc or "calculation" in desc:
                domain = "mathematical"
            elif "file" in desc:
                domain = "file_processing"
            else:
                domain = "general"
            domains[domain] = domains.get(domain, 0) + 1
        
        return {
            "total_mcps": len(all_mcps),
            "avg_success_rate": avg_success_rate,
            "total_executions": total_executions,
            "domains": domains,
            "high_performance_mcps": len(self.get_high_performance_mcps())
        }