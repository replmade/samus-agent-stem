"""MCP (Model Context Protocol) management and generation."""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        """Search for relevant capabilities using enhanced discovery."""
        try:
            from .capability_discovery import CapabilityMatchingEngine
            
            # Create matching engine
            matching_engine = CapabilityMatchingEngine(self.config)
            
            # Convert requirements to task description
            task_description = " ".join(requirements)
            
            # This would be async in a real implementation, for now use sync fallback
            # capabilities = await matching_engine.find_best_capabilities(task_description)
            
            # Return enhanced requirements with discovered capabilities
            enhanced_requirements = requirements.copy()
            
            # Add common capability suggestions based on requirements
            capability_map = {
                "data_processing": ["pandas", "numpy", "csv"],
                "api_integration": ["httpx", "requests", "json"],
                "file_operations": ["pathlib", "aiofiles", "os"],
                "analysis_capability": ["scipy", "matplotlib", "statistics"],
                "general_reasoning": ["json", "logging", "asyncio"]
            }
            
            for req in requirements:
                if req in capability_map:
                    enhanced_requirements.extend(capability_map[req])
            
            return list(set(enhanced_requirements))
            
        except ImportError:
            # Fallback to original behavior
            return requirements
        except Exception as e:
            print(f"Error in capability discovery: {str(e)}")
            return requirements
    
    def _get_fallback_models(self, primary_model: str) -> List[str]:
        """Get fallback models for the primary model."""
        fallback_map = {
            # Anthropic models
            "anthropic/claude-3.5-haiku": ["anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash"],
            "anthropic/claude-3.5-sonnet": ["anthropic/claude-sonnet-4", "anthropic/claude-3.5-haiku"],
            "anthropic/claude-3.7-sonnet": ["anthropic/claude-sonnet-4", "anthropic/claude-3.5-sonnet"],
            "anthropic/claude-sonnet-4": ["anthropic/claude-3.7-sonnet", "anthropic/claude-3.5-sonnet"],
            "anthropic/claude-opus-4": ["anthropic/claude-sonnet-4", "openai/o1"],
            "anthropic/claude-3-opus": ["anthropic/claude-opus-4", "openai/o1"],
            
            # OpenAI models
            "openai/gpt-4o-mini": ["google/gemini-2.0-flash", "anthropic/claude-3.5-haiku"],
            "openai/gpt-4o": ["anthropic/claude-3.5-sonnet", "google/gemini-2.5-pro-preview"],
            "openai/o1-mini": ["anthropic/claude-3.7-sonnet", "google/gemini-2.5-flash-preview"],
            "openai/o1": ["anthropic/claude-opus-4", "openai/o1-mini"],
            "openai/o1-pro": ["openai/o1", "anthropic/claude-opus-4"],
            "openai/o3-mini": ["openai/o1-mini", "anthropic/claude-3.7-sonnet"],
            
            # Google models
            "google/gemini-2.0-flash": ["anthropic/claude-3.5-haiku", "openai/gpt-4o-mini"],
            "google/gemini-2.5-flash-preview": ["anthropic/claude-3.7-sonnet", "openai/o1-mini"],
            "google/gemini-2.5-pro-preview": ["anthropic/claude-sonnet-4", "openai/gpt-4o"],
            "google/gemma-3-27b-it": ["google/gemini-2.0-flash", "anthropic/claude-3.5-haiku"]
        }
        return fallback_map.get(primary_model, ["anthropic/claude-3.5-sonnet"])
    
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
        
        # Import here to avoid circular imports
        from .mcp_codegen import MCPCodeGenerator, MCPValidator
        from .mcp_executor import MCPExecutor
        
        self.code_generator = MCPCodeGenerator(config, model_router)
        self.validator = MCPValidator()
        self.executor = MCPExecutor(config)
    
    async def get_or_create_mcp(self, task_context: str) -> MCPSpecification:
        """Get existing MCP or create new one for task."""
        
        # First, try to find existing relevant MCP
        existing_mcps = self.repository.find_similar_mcps(task_context)
        
        if existing_mcps:
            # Return best matching existing MCP
            return existing_mcps[0]
        
        # Generate new MCP specification
        mcp = self.generator.generate_mcp(task_context)
        
        # Generate the actual code implementation
        generated_code = self.code_generator.generate_mcp_code(mcp)
        
        # Validate the generated code
        validation_result = self.validator.validate_mcp_code(generated_code)
        
        if not validation_result["is_valid"]:
            # Try to sanitize the code
            generated_code = self.validator.sanitize_code(generated_code)
            
            # Re-validate
            validation_result = self.validator.validate_mcp_code(generated_code)
            
            if not validation_result["is_valid"]:
                raise RuntimeError(f"Generated MCP code failed validation: {validation_result}")
        
        # Save MCP files to filesystem
        mcp_path = self.code_generator.save_mcp_files(mcp, generated_code)
        
        # Store specification in repository
        self.repository.store_mcp(mcp)
        
        return mcp
    
    async def execute_mcp(self, mcp: MCPSpecification, input_data: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute an MCP with given input."""
        return await self.executor.execute_mcp(mcp, input_data, parameters)
    
    async def shutdown(self) -> None:
        """Shutdown all MCP processes."""
        await self.executor.shutdown_all_mcps()


class MCPRepository:
    """Manages MCP storage and retrieval with caching and optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.storage_path = Path(config.mcp_repository_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequently accessed MCPs
        self._mcp_cache: Dict[str, MCPSpecification] = {}
        self._cache_access_count: Dict[str, int] = {}
        self._max_cache_size = config.max_mcps_in_memory
        
        # Performance metadata cache
        self._performance_cache: Dict[str, Dict] = {}
        
        # Initialize cache
        self._initialize_cache()
    
    def _initialize_cache(self) -> None:
        """Initialize cache with high-performance MCPs."""
        try:
            # Load high-performance MCPs into cache
            all_mcps = self._load_all_mcps_from_disk()
            
            # Sort by performance metrics
            high_performance_mcps = [
                mcp for mcp in all_mcps 
                if mcp.performance_metrics.get("success_rate", 0.0) > 0.7
            ]
            
            # Cache the top performers
            for mcp in high_performance_mcps[:self._max_cache_size]:
                self._mcp_cache[mcp.mcp_id] = mcp
                self._cache_access_count[mcp.mcp_id] = 0
        
        except Exception as e:
            print(f"Warning: Failed to initialize MCP cache: {str(e)}")
    
    def _load_all_mcps_from_disk(self) -> List[MCPSpecification]:
        """Load all MCPs from disk without caching."""
        mcps = []
        
        if not self.storage_path.exists():
            return mcps
        
        for mcp_file in self.storage_path.glob("*.json"):
            try:
                with open(mcp_file, 'r') as f:
                    data = json.load(f)
                mcp = MCPSpecification(**data)
                mcps.append(mcp)
            except Exception:
                continue  # Skip corrupted files
        
        return mcps
    
    def _update_cache(self, mcp: MCPSpecification) -> None:
        """Update cache with new or modified MCP."""
        # If cache is full, remove least accessed MCP
        if len(self._mcp_cache) >= self._max_cache_size and mcp.mcp_id not in self._mcp_cache:
            if self._cache_access_count:
                least_accessed = min(self._cache_access_count, key=self._cache_access_count.get)
                del self._mcp_cache[least_accessed]
                del self._cache_access_count[least_accessed]
        
        # Add or update MCP in cache
        self._mcp_cache[mcp.mcp_id] = mcp
        if mcp.mcp_id not in self._cache_access_count:
            self._cache_access_count[mcp.mcp_id] = 0
    
    def store_mcp(self, mcp: MCPSpecification) -> None:
        """Store MCP to filesystem and update cache."""
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
        
        # Update cache
        self._update_cache(mcp)
    
    def find_similar_mcps(self, task_context: str, similarity_threshold: float = 0.7, use_semantic: bool = False) -> List[MCPSpecification]:
        """Find MCPs similar to the given task context using semantic similarity."""
        
        # Use fallback by default to avoid model download delays
        if not use_semantic:
            return self._find_similar_mcps_fallback(task_context)
            
        try:
            from .similarity import SemanticSimilarityEngine
            
            # Initialize similarity engine
            similarity_engine = SemanticSimilarityEngine()
            
            # Get all stored MCPs
            all_mcps = self.get_all_mcps()
            
            if not all_mcps:
                return []
            
            # Calculate similarity scores
            similar_mcps = []
            for mcp in all_mcps:
                # Create MCP text representation for similarity comparison
                mcp_text = f"{mcp.name} {mcp.description} {' '.join(mcp.requirements.get('context_requirements', []))}"
                
                # Calculate semantic similarity
                similarity_score = similarity_engine.calculate_similarity(task_context, mcp_text)
                
                if similarity_score >= similarity_threshold:
                    similar_mcps.append((mcp, similarity_score))
            
            # Sort by similarity score (highest first)
            similar_mcps.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 5 most similar MCPs
            return [mcp for mcp, score in similar_mcps[:5]]
            
        except ImportError:
            # Fallback to simple keyword matching if similarity engine not available
            return self._find_similar_mcps_fallback(task_context)
        except Exception as e:
            print(f"Error in semantic similarity search: {str(e)}")
            return self._find_similar_mcps_fallback(task_context)
    
    def _find_similar_mcps_fallback(self, task_context: str) -> List[MCPSpecification]:
        """Fallback similarity search using keyword matching."""
        all_mcps = self.get_all_mcps()
        task_keywords = set(task_context.lower().split())
        
        similar_mcps = []
        for mcp in all_mcps:
            mcp_text = f"{mcp.name} {mcp.description} {' '.join(mcp.requirements.get('context_requirements', []))}"
            mcp_keywords = set(mcp_text.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(task_keywords.intersection(mcp_keywords))
            union = len(task_keywords.union(mcp_keywords))
            
            if union > 0:
                similarity = intersection / union
                if similarity >= 0.2:  # Lower threshold for keyword matching
                    similar_mcps.append((mcp, similarity))
        
        # Sort by similarity score
        similar_mcps.sort(key=lambda x: x[1], reverse=True)
        return [mcp for mcp, score in similar_mcps[:3]]
    
    def load_mcp(self, mcp_id: str) -> Optional[MCPSpecification]:
        """Load MCP by ID, using cache when possible."""
        # Check cache first
        if mcp_id in self._mcp_cache:
            self._cache_access_count[mcp_id] += 1
            return self._mcp_cache[mcp_id]
        
        # Load from disk
        mcp_file = self.storage_path / f"{mcp_id}.json"
        
        if not mcp_file.exists():
            return None
        
        try:
            with open(mcp_file, 'r') as f:
                data = json.load(f)
            
            mcp = MCPSpecification(**data)
            
            # Update cache with loaded MCP
            self._update_cache(mcp)
            
            return mcp
        except Exception as e:
            print(f"Error loading MCP {mcp_id}: {str(e)}")
            return None
    
    def get_all_mcps(self) -> List[MCPSpecification]:
        """Get all stored MCPs with optimized loading."""
        # Start with cached MCPs
        mcps = list(self._mcp_cache.values())
        cached_ids = set(self._mcp_cache.keys())
        
        # Load any MCPs not in cache
        if not self.storage_path.exists():
            return mcps
        
        for mcp_file in self.storage_path.glob("*.json"):
            try:
                mcp_id = mcp_file.stem
                if mcp_id not in cached_ids:
                    with open(mcp_file, 'r') as f:
                        data = json.load(f)
                    mcp = MCPSpecification(**data)
                    mcps.append(mcp)
                    
                    # Consider adding to cache if high performance
                    if mcp.performance_metrics.get("success_rate", 0.0) > 0.7:
                        self._update_cache(mcp)
                        
            except Exception as e:
                print(f"Error loading MCP from {mcp_file}: {str(e)}")
        
        return mcps
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the cache."""
        return {
            "cached_mcps": len(self._mcp_cache),
            "max_cache_size": self._max_cache_size,
            "total_access_count": sum(self._cache_access_count.values())
        }
    
    def clear_cache(self) -> None:
        """Clear the MCP cache."""
        self._mcp_cache.clear()
        self._cache_access_count.clear()
        self._performance_cache.clear()