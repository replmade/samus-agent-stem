"""Agent distillation engine for knowledge transfer between Samus instances."""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .mcp import MCPSpecification, MCPRepository


@dataclass
class DistillationMetrics:
    """Metrics for measuring distillation effectiveness."""
    transfer_success_rate: float
    performance_retention: float
    adaptation_quality: float
    execution_compatibility: float
    cost_efficiency: float
    transfer_time: float


@dataclass
class MCPPerformanceProfile:
    """Comprehensive performance profile for an MCP."""
    mcp_id: str
    name: str
    success_rate: float
    avg_execution_time: float
    cost_per_execution: float
    usage_frequency: int
    complexity_tier: str
    model_compatibility: Dict[str, float]
    domain_effectiveness: Dict[str, float]
    user_satisfaction: float
    error_rate: float
    last_used: float
    created_date: float


class MCPPerformanceAnalyzer:
    """Analyzes and ranks MCP performance for distillation decisions."""
    
    def __init__(self, config: Config):
        self.config = config
        self.repository = MCPRepository(config)
        self.logger = logging.getLogger(__name__)
    
    def analyze_mcp_performance(self, mcp: MCPSpecification) -> MCPPerformanceProfile:
        """Analyze comprehensive performance metrics for an MCP."""
        
        metrics = mcp.performance_metrics
        
        # Calculate derived metrics
        success_rate = metrics.get("success_rate", 0.0)
        avg_execution_time = metrics.get("execution_time", 0.0)
        cost_per_execution = metrics.get("cost_per_execution", 0.0)
        usage_frequency = metrics.get("execution_count", 0)
        
        # Calculate model compatibility scores
        model_efficiency = metrics.get("model_efficiency_scores", {})
        model_compatibility = {}
        for model, scores in model_efficiency.items():
            if isinstance(scores, dict):
                # Weighted score: 40% accuracy, 30% speed, 30% cost
                compatibility = (
                    scores.get("accuracy", 0.0) * 0.4 +
                    (1.0 - scores.get("normalized_latency", 0.5)) * 0.3 +
                    (1.0 - scores.get("normalized_cost", 0.5)) * 0.3
                )
                model_compatibility[model] = compatibility
        
        # Estimate domain effectiveness (placeholder - would be enhanced with actual usage data)
        domain_effectiveness = self._estimate_domain_effectiveness(mcp)
        
        # Calculate user satisfaction (based on success rate and usage frequency)
        user_satisfaction = min(success_rate * (1.0 + min(usage_frequency / 10.0, 1.0)), 1.0)
        
        # Calculate error rate
        error_rate = max(0.0, 1.0 - success_rate)
        
        return MCPPerformanceProfile(
            mcp_id=mcp.mcp_id,
            name=mcp.name,
            success_rate=success_rate,
            avg_execution_time=avg_execution_time,
            cost_per_execution=cost_per_execution,
            usage_frequency=usage_frequency,
            complexity_tier=mcp.model_assignment.get("complexity_tier", "moderate"),
            model_compatibility=model_compatibility,
            domain_effectiveness=domain_effectiveness,
            user_satisfaction=user_satisfaction,
            error_rate=error_rate,
            last_used=metrics.get("last_execution", 0.0),
            created_date=time.time()  # Would be stored in MCP metadata
        )
    
    def _estimate_domain_effectiveness(self, mcp: MCPSpecification) -> Dict[str, float]:
        """Estimate MCP effectiveness across different domains."""
        
        # Analyze MCP description and requirements to categorize domains
        description = mcp.description.lower()
        requirements = mcp.requirements.get("context_requirements", [])
        
        domain_scores = {}
        
        # Domain classification based on keywords
        domain_keywords = {
            "data_analysis": ["data", "analysis", "statistics", "csv", "excel", "pandas"],
            "web_apis": ["http", "api", "request", "web", "rest", "json"],
            "file_processing": ["file", "document", "pdf", "text", "processing"],
            "mathematical": ["math", "calculation", "formula", "algorithm", "compute"],
            "visualization": ["chart", "graph", "plot", "visual", "dashboard"],
            "financial": ["stock", "trading", "finance", "price", "market", "rsi", "macd"]
        }
        
        for domain, keywords in domain_keywords.items():
            score = 0.0
            text_to_analyze = description + " " + " ".join(requirements)
            
            for keyword in keywords:
                if keyword in text_to_analyze:
                    score += 0.2
            
            domain_scores[domain] = min(score, 1.0)
        
        return domain_scores
    
    def rank_mcps_for_distillation(self, mcps: List[MCPSpecification]) -> List[Tuple[MCPSpecification, float]]:
        """Rank MCPs by their suitability for distillation."""
        
        ranked_mcps = []
        
        for mcp in mcps:
            profile = self.analyze_mcp_performance(mcp)
            
            # Calculate distillation score
            distillation_score = self._calculate_distillation_score(profile)
            ranked_mcps.append((mcp, distillation_score))
        
        # Sort by distillation score (highest first)
        ranked_mcps.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_mcps
    
    def _calculate_distillation_score(self, profile: MCPPerformanceProfile) -> float:
        """Calculate a composite score for MCP distillation priority."""
        
        # Weighted scoring formula
        score = (
            profile.success_rate * 0.3 +           # 30% - reliability
            profile.user_satisfaction * 0.25 +     # 25% - user value
            min(profile.usage_frequency / 10.0, 1.0) * 0.2 +  # 20% - popularity (capped)
            (1.0 - profile.error_rate) * 0.15 +    # 15% - stability
            self._calculate_transferability(profile) * 0.1  # 10% - transferability
        )
        
        return score
    
    def _calculate_transferability(self, profile: MCPPerformanceProfile) -> float:
        """Calculate how easily an MCP can be transferred to other models."""
        
        # MCPs with broader model compatibility are more transferable
        if not profile.model_compatibility:
            return 0.5  # Unknown compatibility
        
        avg_compatibility = sum(profile.model_compatibility.values()) / len(profile.model_compatibility)
        
        # MCPs with moderate complexity are often more transferable
        complexity_bonus = {
            "lightweight": 0.9,  # Easy to transfer
            "moderate": 1.0,     # Optimal transferability
            "expert": 0.7        # May require adaptation
        }.get(profile.complexity_tier, 0.8)
        
        return avg_compatibility * complexity_bonus


class ModelCompatibilityAdapter:
    """Adapts MCPs for compatibility with different model configurations."""
    
    def __init__(self, config: Config, model_router):
        self.config = config
        self.model_router = model_router
        self.logger = logging.getLogger(__name__)
    
    def adapt_mcp_for_model(self, mcp: MCPSpecification, target_model: str) -> MCPSpecification:
        """Adapt an MCP for a specific target model."""
        
        # Create a copy of the MCP for adaptation
        adapted_mcp = self._copy_mcp_specification(mcp)
        
        # Update model assignment
        adapted_mcp.model_assignment["default_model"] = target_model
        adapted_mcp.model_assignment["provider"] = self._get_model_provider(target_model)
        
        # Adjust complexity tier based on target model capabilities
        adapted_mcp.model_assignment["complexity_tier"] = self._adjust_complexity_for_model(
            mcp.model_assignment.get("complexity_tier", "moderate"),
            target_model
        )
        
        # Update reasoning requirements
        adapted_mcp.model_assignment["reasoning_requirements"] = self._adjust_reasoning_requirements(
            mcp.model_assignment.get("reasoning_requirements", "advanced"),
            target_model
        )
        
        # Update fallback models
        adapted_mcp.model_assignment["fallback_models"] = self._get_fallback_models_for_target(target_model)
        
        # Adapt model-specific prompts
        adapted_mcp.implementation["model_specific_prompts"] = self._adapt_prompts_for_model(
            mcp.implementation.get("model_specific_prompts", {}),
            target_model
        )
        
        # Reset performance metrics for new model
        adapted_mcp.performance_metrics = {
            "success_rate": 0.0,
            "execution_time": 0.0,
            "resource_usage": "unknown",
            "cost_per_execution": 0.0,
            "model_efficiency_scores": {},
            "transferred_from": mcp.mcp_id,
            "adaptation_date": time.time()
        }
        
        # Update evolution history
        adapted_mcp.evolution_history.append({
            "version": f"{mcp.version}-adapted",
            "changes": f"adapted_for_model_{target_model}",
            "performance_delta": 0.0,
            "model_changes": f"adapted_from_{mcp.model_assignment['default_model']}_to_{target_model}",
            "source_mcp": mcp.mcp_id
        })
        
        return adapted_mcp
    
    def _copy_mcp_specification(self, mcp: MCPSpecification) -> MCPSpecification:
        """Create a deep copy of an MCP specification."""
        import uuid
        
        return MCPSpecification(
            mcp_id=str(uuid.uuid4()),  # New ID for adapted MCP
            version=mcp.version,
            name=f"{mcp.name}_adapted",
            description=f"Adapted version of: {mcp.description}",
            model_assignment=mcp.model_assignment.copy(),
            requirements=mcp.requirements.copy(),
            implementation=mcp.implementation.copy(),
            performance_metrics=mcp.performance_metrics.copy(),
            evolution_history=mcp.evolution_history.copy()
        )
    
    def _get_model_provider(self, model: str) -> str:
        """Determine the provider for a given model."""
        if "anthropic" in model.lower():
            return "anthropic"
        elif "openai" in model.lower() or "gpt" in model.lower():
            return "openai"
        elif "google" in model.lower() or "gemini" in model.lower():
            return "google"
        else:
            return "unknown"
    
    def _adjust_complexity_for_model(self, original_complexity: str, target_model: str) -> str:
        """Adjust complexity tier based on target model capabilities."""
        
        # Model capability mapping (this would be expanded with real benchmarks)
        model_capabilities = {
            "anthropic/claude-3.5-haiku": "lightweight",
            "anthropic/claude-3.5-sonnet": "moderate", 
            "anthropic/claude-3-opus": "expert",
            "openai/gpt-4": "moderate",
            "openai/o1-preview": "expert",
            "google/gemini-pro": "moderate"
        }
        
        target_capability = model_capabilities.get(target_model, "moderate")
        
        # If target model is less capable, keep original complexity
        # If target model is more capable, we can potentially increase complexity
        capability_order = {"lightweight": 0, "moderate": 1, "expert": 2}
        
        original_level = capability_order.get(original_complexity, 1)
        target_level = capability_order.get(target_capability, 1)
        
        if target_level >= original_level:
            return original_complexity
        else:
            return target_capability
    
    def _adjust_reasoning_requirements(self, original_requirements: str, target_model: str) -> str:
        """Adjust reasoning requirements for target model."""
        
        # Similar logic to complexity adjustment
        reasoning_levels = {"basic": 0, "advanced": 1, "expert": 2}
        
        # For now, keep original requirements
        # This could be enhanced with model-specific reasoning benchmarks
        return original_requirements
    
    def _get_fallback_models_for_target(self, target_model: str) -> List[str]:
        """Get appropriate fallback models for the target model."""
        
        fallback_map = {
            "anthropic/claude-3.5-haiku": ["anthropic/claude-3.5-sonnet"],
            "anthropic/claude-3.5-sonnet": ["anthropic/claude-3.5-haiku", "anthropic/claude-3-opus"],
            "anthropic/claude-3-opus": ["anthropic/claude-3.5-sonnet"],
            "openai/gpt-4": ["openai/gpt-3.5-turbo"],
            "google/gemini-pro": ["google/gemini-flash"]
        }
        
        return fallback_map.get(target_model, [])
    
    def _adapt_prompts_for_model(self, original_prompts: Dict[str, str], target_model: str) -> Dict[str, str]:
        """Adapt prompts for the target model's specific characteristics."""
        
        provider = self._get_model_provider(target_model)
        
        # Use existing prompt for the provider if available
        if provider in original_prompts:
            return {provider: original_prompts[provider]}
        
        # Otherwise, adapt the first available prompt
        if original_prompts:
            base_prompt = list(original_prompts.values())[0]
            return {provider: base_prompt}
        
        # Fallback to generic prompt
        return {provider: f"Process this task using {target_model}"}


class AgentDistillationEngine:
    """Core engine for distilling knowledge from source agents to target configurations."""
    
    def __init__(self, config: Config, model_router):
        self.config = config
        self.model_router = model_router
        self.performance_analyzer = MCPPerformanceAnalyzer(config)
        self.compatibility_adapter = ModelCompatibilityAdapter(config, model_router)
        self.logger = logging.getLogger(__name__)
    
    def distill_capabilities(
        self,
        source_agent_mcps: List[MCPSpecification],
        target_model: str,
        performance_threshold: float = 0.7,
        max_mcps: Optional[int] = None
    ) -> Tuple[List[MCPSpecification], DistillationMetrics]:
        """
        Distill capabilities from source agent to target model configuration.
        
        Args:
            source_agent_mcps: List of MCPs from source agent
            target_model: Target model for distillation
            performance_threshold: Minimum performance score for inclusion
            max_mcps: Maximum number of MCPs to distill
            
        Returns:
            Tuple of (distilled_mcps, distillation_metrics)
        """
        
        start_time = time.time()
        
        # Step 1: Analyze and rank MCPs
        ranked_mcps = self.performance_analyzer.rank_mcps_for_distillation(source_agent_mcps)
        
        # Step 2: Filter by performance threshold
        high_performance_mcps = [
            (mcp, score) for mcp, score in ranked_mcps 
            if score >= performance_threshold
        ]
        
        # Step 3: Limit number if specified
        if max_mcps:
            high_performance_mcps = high_performance_mcps[:max_mcps]
        
        # Step 4: Adapt MCPs for target model
        distilled_mcps = []
        adaptation_successes = 0
        
        for mcp, original_score in high_performance_mcps:
            try:
                adapted_mcp = self.compatibility_adapter.adapt_mcp_for_model(mcp, target_model)
                distilled_mcps.append(adapted_mcp)
                adaptation_successes += 1
                
                self.logger.info(f"Successfully adapted MCP {mcp.name} for {target_model}")
                
            except Exception as e:
                self.logger.error(f"Failed to adapt MCP {mcp.name}: {str(e)}")
        
        # Step 5: Calculate distillation metrics
        transfer_time = time.time() - start_time
        
        metrics = DistillationMetrics(
            transfer_success_rate=adaptation_successes / len(high_performance_mcps) if high_performance_mcps else 0.0,
            performance_retention=self._estimate_performance_retention(high_performance_mcps, target_model),
            adaptation_quality=adaptation_successes / len(source_agent_mcps) if source_agent_mcps else 0.0,
            execution_compatibility=0.8,  # Would be measured through actual testing
            cost_efficiency=self._estimate_cost_efficiency(distilled_mcps, target_model),
            transfer_time=transfer_time
        )
        
        self.logger.info(f"Distillation complete: {len(distilled_mcps)} MCPs transferred to {target_model}")
        
        return distilled_mcps, metrics
    
    def _estimate_performance_retention(self, high_performance_mcps: List[Tuple[MCPSpecification, float]], target_model: str) -> float:
        """Estimate how much performance will be retained after transfer."""
        
        if not high_performance_mcps:
            return 0.0
        
        # This would ideally be based on empirical data
        # For now, use heuristics based on model capabilities
        
        total_retention = 0.0
        for mcp, score in high_performance_mcps:
            # Base retention depends on model similarity
            base_retention = 0.8  # Assume 80% retention as baseline
            
            # Adjust based on complexity
            complexity = mcp.model_assignment.get("complexity_tier", "moderate")
            complexity_factor = {
                "lightweight": 0.95,  # Simple MCPs transfer well
                "moderate": 0.85,     # Good transferability
                "expert": 0.75        # May lose some performance
            }.get(complexity, 0.85)
            
            retention = base_retention * complexity_factor
            total_retention += retention
        
        return total_retention / len(high_performance_mcps)
    
    def _estimate_cost_efficiency(self, distilled_mcps: List[MCPSpecification], target_model: str) -> float:
        """Estimate cost efficiency of distilled MCPs with target model."""
        
        # This would be calculated based on actual model pricing
        # For now, return a reasonable estimate
        return 0.85
    
    def export_mcps(self, mcps: List[MCPSpecification], export_path: Path) -> bool:
        """Export MCPs to a portable format for sharing."""
        
        try:
            export_data = {
                "version": "1.0",
                "export_date": time.time(),
                "mcps": []
            }
            
            for mcp in mcps:
                mcp_data = {
                    "mcp_id": mcp.mcp_id,
                    "version": mcp.version,
                    "name": mcp.name,
                    "description": mcp.description,
                    "model_assignment": mcp.model_assignment,
                    "requirements": mcp.requirements,
                    "implementation": mcp.implementation,
                    "performance_metrics": mcp.performance_metrics,
                    "evolution_history": mcp.evolution_history
                }
                export_data["mcps"].append(mcp_data)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(mcps)} MCPs to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export MCPs: {str(e)}")
            return False
    
    def import_mcps(self, import_path: Path) -> List[MCPSpecification]:
        """Import MCPs from exported format."""
        
        try:
            with open(import_path, 'r') as f:
                export_data = json.load(f)
            
            imported_mcps = []
            
            for mcp_data in export_data.get("mcps", []):
                mcp = MCPSpecification(**mcp_data)
                imported_mcps.append(mcp)
            
            self.logger.info(f"Imported {len(imported_mcps)} MCPs from {import_path}")
            return imported_mcps
            
        except Exception as e:
            self.logger.error(f"Failed to import MCPs: {str(e)}")
            return []