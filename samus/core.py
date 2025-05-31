"""Minimal Agent Core implementation."""

from dataclasses import dataclass
from typing import List

from .config import Config


@dataclass
class ProblemSolution:
    """Solution returned by the minimal agent core."""
    content: str
    mcps_used: List[str]
    reasoning_trace: List[str]


class ReasoningEngine:
    """Core reasoning engine using Claude Sonnet 4."""
    
    def __init__(self, config: Config, model_router):
        self.config = config
        self.model_router = model_router
        self.model = config.supervisor_model
    
    def process(self, problem: str, context: str = "") -> ProblemSolution:
        """Process a problem using direct Claude Sonnet 4 reasoning."""
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(problem, context)
        
        try:
            content = self.model_router.call_model(
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
                max_tokens=4096,
                temperature=0.1
            )
            
            return ProblemSolution(
                content=content,
                mcps_used=[],  # Will be populated by MCP manager
                reasoning_trace=[f"Used {self.model} for direct reasoning"]
            )
            
        except Exception as e:
            return ProblemSolution(
                content=f"Reasoning error: {str(e)}",
                mcps_used=[],
                reasoning_trace=[f"Error in reasoning engine: {str(e)}"]
            )
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the reasoning engine."""
        return """You are Samus, an autonomous agent with minimal predefinition and maximal self-evolution capabilities.

Your core principles:
1. Solve problems through direct reasoning rather than predefined tools
2. When you identify capability gaps, describe what MCPs (Model Context Protocols) should be generated
3. Focus on the "why" behind solutions, not just the "what"
4. Evolve your approach based on problem requirements

When you encounter a problem:
1. Analyze the requirements directly using your reasoning capabilities
2. If you need external capabilities, clearly specify what MCP should be generated
3. Provide step-by-step reasoning for your approach
4. Suggest how the solution could be improved or evolved

You have access to dynamic MCP generation - if you need specific capabilities that don't exist, describe them and they will be created for you."""
    
    def _build_user_prompt(self, problem: str, context: str) -> str:
        """Build the user prompt with problem and context."""
        prompt = f"Problem to solve: {problem}"
        
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        prompt += "\n\nPlease provide your reasoning and solution."
        
        return prompt


class ContextManager:
    """Manages context and state for the agent."""
    
    def __init__(self):
        self.conversation_history: List[str] = []
        self.active_mcps: List[str] = []
        self.performance_metrics: dict = {}
    
    def add_interaction(self, problem: str, solution: ProblemSolution) -> None:
        """Add an interaction to the conversation history."""
        self.conversation_history.append(f"Problem: {problem}")
        self.conversation_history.append(f"Solution: {solution.content}")
        self.active_mcps.extend(solution.mcps_used)
    
    def get_context_summary(self) -> str:
        """Get a summary of recent context for problem solving."""
        if not self.conversation_history:
            return ""
        
        # Return last few interactions as context
        recent_history = self.conversation_history[-6:]  # Last 3 problem-solution pairs
        return "\n".join(recent_history)


class MinimalAgentCore:
    """
    Single component responsible for direct problem-solving
    using Claude Sonnet 4's reasoning capabilities.
    """
    
    def __init__(self, config: Config, model_router, mcp_manager):
        self.config = config
        self.model_router = model_router
        self.mcp_manager = mcp_manager
        self.reasoning_engine = ReasoningEngine(config, model_router)
        self.context_manager = ContextManager()
    
    def solve_problem(self, problem: str) -> ProblemSolution:
        """Solve a problem using direct reasoning and dynamic MCP generation."""
        
        # Get context from previous interactions
        context = self.context_manager.get_context_summary()
        
        # First, try direct reasoning
        solution = self.reasoning_engine.process(problem, context)
        
        # TODO: Analyze solution for MCP generation needs
        # This will be implemented when MCP framework is ready
        
        # Add interaction to context
        self.context_manager.add_interaction(problem, solution)
        
        return solution