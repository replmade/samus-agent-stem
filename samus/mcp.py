"""MCP (Model Context Protocol) management and generation."""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config


@dataclass
class MCPSpecification:
    """MCP specification with model assignment."""
    mcp_id: str
    version: str
    name: str
    description: str
    model_assignment: Dict
    requirements: Dict
    implementation: Dict
    performance_metrics: Dict
    evolution_history: List[Dict]


class MCPGenerator:
    """Dynamically generates Model Context Protocols with intelligent model assignment."""
    
    def __init__(self, config: Config, model_router):
        self.config = config
        self.model_router = model_router
        self.complexity_analyzer = TaskComplexityAnalyzer()
    
    def generate_mcp(self, task_context: str) -> MCPSpecification:
        """Generate an MCP for the given task context."""
        
        # Analyze task requirements and complexity
        requirements = self._analyze_requirements(task_context)
        complexity = self.complexity_analyzer.assess_task_complexity(task_context)
        
        # Search for relevant capabilities (placeholder for now)
        capabilities = self._search_capabilities(requirements)
        
        # Assign optimal model based on capability requirements
        default_model = self.model_router.select_optimal_model(
            capabilities, complexity, requirements
        )
        
        # Generate MCP specification
        mcp_id = str(uuid.uuid4())
        
        return MCPSpecification(
            mcp_id=mcp_id,
            version="1.0.0",
            name=f"dynamic_capability_{mcp_id[:8]}",
            description=f"Generated capability for: {task_context[:100]}",
            model_assignment={
                "default_model": default_model,
                "provider": "anthropic",  # Default for now
                "complexity_tier": complexity,
                "cost_optimization": True,
                "fallback_models": self._get_fallback_models(default_model),
                "reasoning_requirements": self._get_reasoning_requirements(complexity),
                "performance_profile": {
                    "avg_tokens": 150,
                    "avg_latency_ms": 800,
                    "success_rate_by_model": {}
                }
            },
            requirements={
                "input_format": "text",
                "output_format": "text",
                "context_requirements": requirements,
                "computational_complexity": complexity,
                "reasoning_depth": self._map_complexity_to_reasoning(complexity)
            },
            implementation={
                "protocol_steps": self._generate_protocol_steps(task_context),
                "resource_endpoints": [],
                "validation_rules": [],
                "model_specific_prompts": {
                    "anthropic": f"Process this task: {task_context}",
                    "openai": f"Handle this request: {task_context}"
                }
            },
            performance_metrics={
                "success_rate": 0.0,
                "execution_time": 0.0,
                "resource_usage": "unknown",
                "cost_per_execution": 0.0,
                "model_efficiency_scores": {}
            },
            evolution_history=[{
                "version": "1.0.0",
                "changes": "initial_implementation",
                "performance_delta": 0.0,
                "model_changes": f"assigned_default_{default_model}"
            }]
        )
    
    def refine_mcp(self, mcp: MCPSpecification, feedback: Dict) -> MCPSpecification:
        """Evolve MCP based on execution feedback."""
        # TODO: Implement MCP evolution logic
        return mcp
    
    def _analyze_requirements(self, task_context: str) -> List[str]:
        """Extract requirements from task context."""
        # Simple keyword-based analysis for now
        requirements = []
        
        if "file" in task_context.lower():
            requirements.append("file_operations")
        if "data" in task_context.lower():
            requirements.append("data_processing")
        if "api" in task_context.lower():
            requirements.append("api_integration")
        if "analysis" in task_context.lower():
            requirements.append("analysis_capability")
        
        return requirements or ["general_reasoning"]
    
    def _search_capabilities(self, requirements: List[str]) -> List[str]:
        """Search for relevant capabilities (placeholder)."""
        # TODO: Implement actual capability discovery
        return requirements
    
    def _get_fallback_models(self, primary_model: str) -> List[str]:
        """Get fallback models for the primary model."""
        fallback_map = {
            "claude-3-5-haiku": ["claude-sonnet-4"],
            "claude-sonnet-4": ["claude-3-5-haiku", "claude-opus-4"],
            "claude-opus-4": ["claude-sonnet-4"]
        }
        return fallback_map.get(primary_model, [])
    
    def _get_reasoning_requirements(self, complexity: str) -> str:
        """Map complexity to reasoning requirements."""
        return {
            "lightweight": "basic",
            "moderate": "advanced", 
            "expert": "expert"
        }.get(complexity, "basic")
    
    def _map_complexity_to_reasoning(self, complexity: str) -> str:
        """Map complexity to reasoning depth."""
        return {
            "lightweight": "basic",
            "moderate": "advanced",
            "expert": "expert"
        }.get(complexity, "basic")
    
    def _generate_protocol_steps(self, task_context: str) -> List[str]:
        """Generate protocol steps for the task."""
        return [
            "analyze_input",
            "process_requirements", 
            "generate_solution",
            "validate_output"
        ]


class TaskComplexityAnalyzer:
    """Analyzes task complexity to determine appropriate model assignment."""
    
    def __init__(self):
        self.complexity_indicators = {
            "lightweight": ["file_ops", "data_format", "simple_api", "basic_math"],
            "moderate": ["text_analysis", "code_generation", "api_orchestration"],
            "expert": ["complex_reasoning", "multi_step_analysis", "research_synthesis"]
        }
    
    def assess_task_complexity(self, task: str) -> str:
        """Assess complexity based on task characteristics."""
        task_lower = task.lower()
        task_keywords = task_lower.split()
        
        complexity_scores = {
            "lightweight": self._calculate_match_score(task_keywords, "lightweight"),
            "moderate": self._calculate_match_score(task_keywords, "moderate"),
            "expert": self._calculate_match_score(task_keywords, "expert")
        }
        
        # Default to moderate if no clear indicators
        max_complexity = max(complexity_scores, key=complexity_scores.get)
        return max_complexity if complexity_scores[max_complexity] > 0 else "moderate"
    
    def _calculate_match_score(self, task_keywords: List[str], complexity: str) -> float:
        """Calculate match score for complexity level."""
        indicators = self.complexity_indicators[complexity]
        matches = sum(1 for keyword in task_keywords if any(ind in keyword for ind in indicators))
        return matches / len(task_keywords) if task_keywords else 0


class MCPManager:
    """Manages MCP lifecycle, storage, and retrieval."""
    
    def __init__(self, config: Config, model_router):
        self.config = config
        self.model_router = model_router
        self.generator = MCPGenerator(config, model_router)
        self.repository = MCPRepository(config)
    
    def get_or_create_mcp(self, task_context: str) -> MCPSpecification:
        """Get existing MCP or create new one for task."""
        
        # First, try to find existing relevant MCP
        existing_mcps = self.repository.find_similar_mcps(task_context)
        
        if existing_mcps:
            # Return best matching existing MCP
            return existing_mcps[0]
        
        # Generate new MCP
        mcp = self.generator.generate_mcp(task_context)
        
        # Store for future use
        self.repository.store_mcp(mcp)
        
        return mcp
    
    def execute_mcp(self, mcp: MCPSpecification, input_data: str) -> str:
        """Execute an MCP with given input."""
        # TODO: Implement actual MCP execution
        # For now, just use the model router to call the assigned model
        
        messages = [{"role": "user", "content": input_data}]
        system_prompt = mcp.implementation["model_specific_prompts"].get("anthropic", "")
        
        return self.model_router.call_model(
            mcp.model_assignment["default_model"],
            messages,
            system_prompt
        )


class MCPRepository:
    """Manages MCP storage and retrieval."""
    
    def __init__(self, config: Config):
        self.config = config
        self.storage_path = Path(config.mcp_repository_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store_mcp(self, mcp: MCPSpecification) -> None:
        """Store MCP to filesystem."""
        mcp_file = self.storage_path / f"{mcp.mcp_id}.json"
        
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
        
        with open(mcp_file, 'w') as f:
            json.dump(mcp_data, f, indent=2)
    
    def find_similar_mcps(self, task_context: str) -> List[MCPSpecification]:
        """Find MCPs similar to the given task context."""
        # TODO: Implement semantic similarity search
        # For now, return empty list to force generation
        return []
    
    def load_mcp(self, mcp_id: str) -> Optional[MCPSpecification]:
        """Load MCP by ID."""
        mcp_file = self.storage_path / f"{mcp_id}.json"
        
        if not mcp_file.exists():
            return None
        
        with open(mcp_file, 'r') as f:
            data = json.load(f)
        
        return MCPSpecification(**data)