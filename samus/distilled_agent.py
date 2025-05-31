"""DistilledAgent class for agents with inherited capabilities."""

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .agent import SamusAgent, AgentResponse
from .config import Config
from .core import MinimalAgentCore
from .distillation import AgentDistillationEngine, DistillationMetrics, MCPPerformanceProfile
from .mcp import MCPManager, MCPSpecification
from .models import ModelRouter


@dataclass
class DistillationInfo:
    """Information about the distillation process for this agent."""
    source_agent_id: Optional[str]
    distillation_date: float
    target_model: str
    inherited_mcps_count: int
    distillation_metrics: DistillationMetrics
    performance_baseline: Dict[str, float]


class DistilledAgent(SamusAgent):
    """
    A Samus agent that inherits capabilities from other agent instances.
    
    This agent starts with pre-distilled MCPs from successful parent agents,
    giving it inherited knowledge while maintaining the ability to evolve new capabilities.
    """
    
    def __init__(
        self, 
        config: Config, 
        inherited_mcps: List[MCPSpecification] = None,
        distillation_info: DistillationInfo = None
    ):
        # Initialize base agent
        super().__init__(config)
        
        # Store distillation information
        self.distillation_info = distillation_info
        self.inherited_mcps = inherited_mcps or []
        self.inherited_mcp_ids = {mcp.mcp_id for mcp in self.inherited_mcps}
        
        # Track performance of inherited vs generated MCPs
        self.performance_tracking = {
            "inherited_mcp_usage": 0,
            "generated_mcp_usage": 0,
            "inherited_success_rate": 0.0,
            "generated_success_rate": 0.0,
            "capability_evolution_rate": 0.0
        }
        
        # Install inherited MCPs
        if self.inherited_mcps:
            self._install_inherited_mcps()
    
    def _install_inherited_mcps(self) -> None:
        """Install inherited MCPs into the agent's capability repository."""
        
        installed_count = 0
        
        for mcp in self.inherited_mcps:
            try:
                # Store the inherited MCP in the repository
                self.mcp_manager.repository.store_mcp(mcp)
                
                # Mark as inherited in performance metrics
                mcp.performance_metrics["inherited"] = True
                mcp.performance_metrics["inheritance_date"] = time.time()
                
                installed_count += 1
                
            except Exception as e:
                print(f"Failed to install inherited MCP {mcp.name}: {str(e)}")
        
        print(f"Successfully installed {installed_count} inherited MCPs")
    
    async def process(self, prompt: str) -> AgentResponse:
        """
        Process a prompt with preference for inherited capabilities when applicable.
        """
        start_time = time.time()
        
        try:
            # Check if we have relevant inherited MCPs for this task
            relevant_inherited = await self._find_relevant_inherited_mcps(prompt)
            
            # Process using the core with inherited context
            result = await self.core.solve_problem(prompt)
            
            # Track usage of inherited vs generated MCPs
            self._update_usage_tracking(result)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                content=result.content,
                mcps_used=result.mcps_used,
                execution_time=execution_time,
                reasoning_trace=result.reasoning_trace + self._get_inheritance_trace(),
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResponse(
                content=f"Error processing request: {str(e)}",
                mcps_used=[],
                execution_time=execution_time,
                reasoning_trace=[f"Error: {str(e)}"],
            )
    
    async def _find_relevant_inherited_mcps(self, prompt: str) -> List[MCPSpecification]:
        """Find inherited MCPs that might be relevant to the current prompt."""
        
        relevant_mcps = []
        prompt_lower = prompt.lower()
        
        for mcp in self.inherited_mcps:
            # Simple relevance scoring based on description and requirements
            relevance_score = 0.0
            
            # Check description overlap
            description_words = mcp.description.lower().split()
            for word in description_words:
                if word in prompt_lower:
                    relevance_score += 0.2
            
            # Check requirements overlap
            requirements = mcp.requirements.get("context_requirements", [])
            for req in requirements:
                if req.lower() in prompt_lower:
                    relevance_score += 0.3
            
            # Include if relevance score is above threshold
            if relevance_score > 0.5:
                relevant_mcps.append(mcp)
        
        return relevant_mcps
    
    def _update_usage_tracking(self, result) -> None:
        """Update tracking of inherited vs generated MCP usage."""
        
        for mcp_id in result.mcps_used:
            if mcp_id in self.inherited_mcp_ids:
                self.performance_tracking["inherited_mcp_usage"] += 1
            else:
                self.performance_tracking["generated_mcp_usage"] += 1
    
    def _get_inheritance_trace(self) -> List[str]:
        """Get tracing information about inheritance."""
        
        if not self.distillation_info:
            return []
        
        trace = [
            f"Agent inherited {len(self.inherited_mcps)} capabilities from source agent",
            f"Distillation success rate: {self.distillation_info.distillation_metrics.transfer_success_rate:.2f}"
        ]
        
        return trace
    
    def get_capability_summary(self) -> Dict[str, Any]:
        """Get a summary of inherited and generated capabilities."""
        
        total_mcps = len(self.inherited_mcps)
        
        # Get generated MCPs from repository
        all_stored_mcps = self.mcp_manager.repository.get_all_mcps()
        generated_mcps = [mcp for mcp in all_stored_mcps if mcp.mcp_id not in self.inherited_mcp_ids]
        
        return {
            "inherited_capabilities": {
                "count": len(self.inherited_mcps),
                "domains": self._categorize_mcps_by_domain(self.inherited_mcps),
                "avg_performance": self._calculate_avg_performance(self.inherited_mcps)
            },
            "generated_capabilities": {
                "count": len(generated_mcps),
                "domains": self._categorize_mcps_by_domain(generated_mcps),
                "avg_performance": self._calculate_avg_performance(generated_mcps)
            },
            "usage_statistics": self.performance_tracking,
            "distillation_info": {
                "source_model": self.distillation_info.target_model if self.distillation_info else None,
                "distillation_date": self.distillation_info.distillation_date if self.distillation_info else None,
                "transfer_effectiveness": self.distillation_info.distillation_metrics.transfer_success_rate if self.distillation_info else None
            }
        }
    
    def _categorize_mcps_by_domain(self, mcps: List[MCPSpecification]) -> Dict[str, int]:
        """Categorize MCPs by their domain/purpose."""
        
        domains = {}
        
        for mcp in mcps:
            # Simple categorization based on description keywords
            description = mcp.description.lower()
            
            if any(word in description for word in ["data", "analysis", "csv", "excel"]):
                domain = "data_analysis"
            elif any(word in description for word in ["http", "api", "web", "request"]):
                domain = "web_apis"
            elif any(word in description for word in ["math", "calculation", "formula"]):
                domain = "mathematical"
            elif any(word in description for word in ["file", "document", "text"]):
                domain = "file_processing"
            elif any(word in description for word in ["chart", "graph", "visual"]):
                domain = "visualization"
            elif any(word in description for word in ["stock", "financial", "trading"]):
                domain = "financial"
            else:
                domain = "general"
            
            domains[domain] = domains.get(domain, 0) + 1
        
        return domains
    
    def _calculate_avg_performance(self, mcps: List[MCPSpecification]) -> float:
        """Calculate average performance score for a list of MCPs."""
        
        if not mcps:
            return 0.0
        
        total_score = 0.0
        valid_mcps = 0
        
        for mcp in mcps:
            success_rate = mcp.performance_metrics.get("success_rate", 0.0)
            if success_rate > 0:
                total_score += success_rate
                valid_mcps += 1
        
        return total_score / valid_mcps if valid_mcps > 0 else 0.0
    
    async def evolve_inherited_capability(self, mcp_id: str, feedback: Dict[str, Any]) -> bool:
        """
        Evolve an inherited capability based on usage feedback.
        
        This allows inherited MCPs to continue evolving in the new agent context.
        """
        
        if mcp_id not in self.inherited_mcp_ids:
            return False
        
        try:
            # Find the inherited MCP
            inherited_mcp = next(mcp for mcp in self.inherited_mcps if mcp.mcp_id == mcp_id)
            
            # Apply evolution using the MCP manager
            evolved_mcp = self.mcp_manager.generator.refine_mcp(inherited_mcp, feedback)
            
            # Update the inherited MCP list
            for i, mcp in enumerate(self.inherited_mcps):
                if mcp.mcp_id == mcp_id:
                    self.inherited_mcps[i] = evolved_mcp
                    break
            
            # Store the evolved version
            self.mcp_manager.repository.store_mcp(evolved_mcp)
            
            # Update performance tracking
            self.performance_tracking["capability_evolution_rate"] += 1
            
            return True
            
        except Exception as e:
            print(f"Failed to evolve inherited capability {mcp_id}: {str(e)}")
            return False
    
    def compare_with_source_performance(self) -> Dict[str, float]:
        """
        Compare current performance with the source agent's baseline.
        
        This helps measure if the distilled agent is maintaining or improving
        upon the inherited capabilities.
        """
        
        if not self.distillation_info or not self.distillation_info.performance_baseline:
            return {}
        
        current_performance = {}
        baseline_performance = self.distillation_info.performance_baseline
        
        # Calculate current performance metrics
        for mcp in self.inherited_mcps:
            mcp_id = mcp.mcp_id
            current_success_rate = mcp.performance_metrics.get("success_rate", 0.0)
            baseline_success_rate = baseline_performance.get(mcp_id, 0.0)
            
            if baseline_success_rate > 0:
                performance_ratio = current_success_rate / baseline_success_rate
                current_performance[mcp_id] = performance_ratio
        
        return current_performance
    
    @classmethod
    def create_from_distillation(
        cls,
        config: Config,
        source_mcps: List[MCPSpecification],
        target_model: str,
        source_agent_id: Optional[str] = None,
        performance_threshold: float = 0.7,
        max_mcps: Optional[int] = None
    ) -> "DistilledAgent":
        """
        Create a DistilledAgent by distilling capabilities from source MCPs.
        
        Args:
            config: Configuration for the new agent
            source_mcps: MCPs from the source agent
            target_model: Target model for the distilled agent
            source_agent_id: ID of the source agent (for tracking)
            performance_threshold: Minimum performance for MCP inclusion
            max_mcps: Maximum number of MCPs to distill
            
        Returns:
            DistilledAgent with inherited capabilities
        """
        
        # Create model router for the target configuration
        model_router = ModelRouter(config)
        
        # Create distillation engine
        distillation_engine = AgentDistillationEngine(config, model_router)
        
        # Perform distillation
        distilled_mcps, metrics = distillation_engine.distill_capabilities(
            source_mcps,
            target_model,
            performance_threshold,
            max_mcps
        )
        
        # Create performance baseline
        performance_baseline = {}
        for mcp in source_mcps:
            performance_baseline[mcp.mcp_id] = mcp.performance_metrics.get("success_rate", 0.0)
        
        # Create distillation info
        distillation_info = DistillationInfo(
            source_agent_id=source_agent_id,
            distillation_date=time.time(),
            target_model=target_model,
            inherited_mcps_count=len(distilled_mcps),
            distillation_metrics=metrics,
            performance_baseline=performance_baseline
        )
        
        # Create the distilled agent
        return cls(config, distilled_mcps, distillation_info)