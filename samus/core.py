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
    
    async def solve_problem(self, problem: str) -> ProblemSolution:
        """Solve a problem using direct reasoning and dynamic MCP generation."""
        
        # Get context from previous interactions
        context = self.context_manager.get_context_summary()
        
        # First, try direct reasoning
        solution = self.reasoning_engine.process(problem, context)
        
        # Analyze if we need to generate and execute MCPs (if enabled)
        if self.config.enable_mcp_generation:
            mcp_needs = await self._analyze_mcp_needs(problem, solution)
            
            if mcp_needs:
                # Generate and execute MCPs to enhance the solution
                enhanced_solution = await self._execute_mcps(problem, solution, mcp_needs)
                solution = enhanced_solution
        else:
            solution.reasoning_trace.append("MCP generation disabled - using direct reasoning only")
        
        # Add interaction to context
        self.context_manager.add_interaction(problem, solution)
        
        return solution
    
    async def _analyze_mcp_needs(self, problem: str, solution: ProblemSolution) -> List[str]:
        """Analyze if MCPs are needed to enhance the solution."""
        
        # Use the reasoning engine to determine if MCPs are needed
        analysis_prompt = f"""Given this problem and initial solution, determine if the user is requesting actual executable MCP generation.

Problem: {problem}

Initial Solution: {solution.content}

CRITICAL: Look for these indicators that the user wants executable MCP generation:
1. Explicit words: "Generate", "Create", "Build" + "MCP server" 
2. Executable requirements: "executable", "run immediately", "that I can run"
3. File/code generation requests: "Python MCP server", "MCP implementation"
4. NOT just asking for explanations, concepts, or descriptions

If the user is requesting actual MCP generation/creation, you MUST set needs_mcps to true.

Additional MCP considerations if needs_mcps is true:
1. Does this require external API integration?
2. Does this need specialized data processing capabilities?
3. Would separate validation/caching MCPs improve reliability?
4. Are there security/authentication requirements?

Respond with JSON format:
{{
    "needs_mcps": true/false,
    "mcp_requirements": ["description1", "description2"],
    "reasoning": "explanation"
}}"""
        
        response = self.reasoning_engine.model_router.call_model(
            model=self.reasoning_engine.model,
            messages=[{"role": "user", "content": analysis_prompt}],
            system="You are an expert at identifying when specialized capabilities are needed.",
            max_tokens=1024,
            temperature=0.1
        )
        
        try:
            import json
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()
            
            analysis = json.loads(json_str)
            if analysis.get("needs_mcps", False):
                requirements = analysis.get("mcp_requirements", [])
                return requirements
        except json.JSONDecodeError as e:
            pass  # If we can't parse the response, assume no MCPs needed
        
        return []
    
    async def _execute_mcps(self, problem: str, solution: ProblemSolution, mcp_needs: List[str]) -> ProblemSolution:
        """Execute MCPs to enhance the solution with timeout handling."""
        import asyncio
        
        enhanced_content = solution.content
        mcps_used = []
        reasoning_trace = solution.reasoning_trace.copy()
        
        for mcp_requirement in mcp_needs:
            try:
                # Generate or retrieve MCP for this requirement with timeout
                mcp = await asyncio.wait_for(
                    self.mcp_manager.get_or_create_mcp(mcp_requirement),
                    timeout=self.config.mcp_generation_timeout
                )
                
                # Execute the MCP with the problem context with timeout
                mcp_result = await asyncio.wait_for(
                    self.mcp_manager.execute_mcp(
                        mcp, 
                        f"Problem: {problem}\nCurrent Solution: {enhanced_content}",
                        {"requirement": mcp_requirement}
                    ),
                    timeout=self.config.mcp_execution_timeout
                )
                
                if mcp_result["success"]:
                    # Integrate MCP result into the solution
                    integration_prompt = f"""Integrate this MCP result into the existing solution:

Original Solution: {enhanced_content}

MCP Result: {mcp_result['result']}

MCP Purpose: {mcp_requirement}

Provide an enhanced, integrated solution:"""
                    
                    enhanced_content = self.reasoning_engine.model_router.call_model(
                        model=self.reasoning_engine.model,
                        messages=[{"role": "user", "content": integration_prompt}],
                        system="Integrate the MCP results seamlessly into the solution.",
                        max_tokens=4096,
                        temperature=0.1
                    )
                    
                    mcps_used.append(mcp.mcp_id)
                    reasoning_trace.append(f"Enhanced with MCP: {mcp.name}")
                
            except asyncio.TimeoutError:
                reasoning_trace.append(f"MCP generation/execution timed out for '{mcp_requirement}' - skipping")
                print(f"Warning: MCP timeout for requirement: {mcp_requirement}")
            except Exception as e:
                reasoning_trace.append(f"MCP execution failed for '{mcp_requirement}': {str(e)}")
                print(f"Warning: MCP error for requirement: {mcp_requirement} - {str(e)}")
        
        return ProblemSolution(
            content=enhanced_content,
            mcps_used=mcps_used,
            reasoning_trace=reasoning_trace
        )