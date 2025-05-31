"""Dynamic MCP code generation from specifications."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .mcp import MCPSpecification


class MCPCodeGenerator:
    """Generates executable MCP servers from specifications."""
    
    def __init__(self, config: Config, model_router):
        self.config = config
        self.model_router = model_router
        self.template_generator = MCPTemplateGenerator(model_router)
    
    def generate_mcp_code(self, spec: MCPSpecification) -> str:
        """Generate complete MCP server code from specification."""
        
        # Use the assigned model to generate the actual implementation
        assigned_model = spec.model_assignment["default_model"]
        
        generation_prompt = self._build_generation_prompt(spec)
        
        generated_code = self.model_router.call_model(
            model=assigned_model,
            messages=[{"role": "user", "content": generation_prompt}],
            system=self._build_system_prompt(),
            max_tokens=8192,
            temperature=0.1
        )
        
        # Clean up any markdown code blocks that might be embedded
        if "```python" in generated_code:
            start = generated_code.find("```python") + 9
            end = generated_code.find("```", start)
            if end != -1:
                generated_code = generated_code[start:end].strip()
        elif "```" in generated_code:
            start = generated_code.find("```") + 3
            end = generated_code.find("```", start)  
            if end != -1:
                generated_code = generated_code[start:end].strip()
        
        # Wrap the generated implementation in the MCP server template
        return self.template_generator.create_mcp_server(spec, generated_code)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for MCP code generation."""
        return """You are an expert MCP (Model Context Protocol) server generator. Your task is to create robust, production-ready Python code that implements the requested capability.

Key requirements:
1. Generate only the core implementation functions - NOT the full MCP server boilerplate
2. Use proper error handling and logging
3. Include type hints and docstrings
4. Follow security best practices
5. Make the code modular and testable
6. Use async/await where appropriate for I/O operations
7. Include proper validation of inputs and outputs

The generated code will be wrapped in an MCP server template, so focus on the business logic implementation."""
    
    def _build_generation_prompt(self, spec: MCPSpecification) -> str:
        """Build the prompt for generating MCP implementation."""
        prompt = f"""Generate Python implementation for this MCP capability:

**MCP Specification:**
- Name: {spec.name}
- Description: {spec.description}
- Complexity: {spec.model_assignment.get('complexity_tier', 'moderate')}

**Requirements:**
{json.dumps(spec.requirements, indent=2)}

**Protocol Steps:**
{json.dumps(spec.implementation.get('protocol_steps', []), indent=2)}

**Resource Endpoints:**
{json.dumps(spec.implementation.get('resource_endpoints', []), indent=2)}

**CRITICAL REQUIREMENTS:**
1. You MUST include a function named 'async def execute_capability(input_data: str, parameters: dict) -> dict'
2. DO NOT use exec(), eval(), compile(), __import__(), or subprocess
3. Use only safe imports: httpx, aiofiles, json, asyncio, logging, pathlib
4. Include proper error handling with try/catch blocks
5. Add type hints and docstrings
6. Use async/await for I/O operations
7. Return results as a dictionary

**Example Structure:**
```python
import asyncio
import httpx
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def execute_capability(input_data: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Main execution function for this MCP capability.\"\"\"
    try:
        # Your implementation here
        result = {{
            "success": True,
            "data": "your_result_here"
        }}
        return result
    except Exception as e:
        logger.error(f"Error in execute_capability: {{str(e)}}")
        return {{
            "success": False,
            "error": str(e)
        }}
```

Generate ONLY the implementation code (no explanations):"""
        
        return prompt
    
    def create_mcp_directory(self, spec: MCPSpecification) -> Path:
        """Create directory structure for the MCP."""
        mcp_path = Path(self.config.mcp_repository_path) / spec.mcp_id
        mcp_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (mcp_path / "src").mkdir(exist_ok=True)
        (mcp_path / "tests").mkdir(exist_ok=True)
        (mcp_path / "logs").mkdir(exist_ok=True)
        
        return mcp_path
    
    def save_mcp_files(self, spec: MCPSpecification, code: str) -> Path:
        """Save MCP files to filesystem."""
        mcp_path = self.create_mcp_directory(spec)
        
        # Save the main server file
        server_file = mcp_path / "server.py"
        with open(server_file, 'w') as f:
            f.write(code)
        
        # Save the specification
        spec_file = mcp_path / "specification.json"
        with open(spec_file, 'w') as f:
            json.dump({
                "mcp_id": spec.mcp_id,
                "version": spec.version,
                "name": spec.name,
                "description": spec.description,
                "model_assignment": spec.model_assignment,
                "requirements": spec.requirements,
                "implementation": spec.implementation,
                "performance_metrics": spec.performance_metrics,
                "evolution_history": spec.evolution_history
            }, f, indent=2)
        
        # Create requirements.txt for the MCP
        requirements_file = mcp_path / "requirements.txt"
        requirements = self._generate_requirements(spec)
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(requirements))
        
        # Create startup script
        startup_script = mcp_path / "start.sh"
        with open(startup_script, 'w') as f:
            f.write(f"""#!/bin/bash
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python server.py
""")
        startup_script.chmod(0o755)
        
        return mcp_path
    
    def _generate_requirements(self, spec: MCPSpecification) -> List[str]:
        """Generate requirements.txt content based on MCP specification."""
        base_requirements = [
            "mcp>=1.0.0",
            "pydantic>=2.5.0",
            "httpx>=0.25.0",
            "aiofiles>=23.0.0"
        ]
        
        # Add specific requirements based on MCP type
        requirements = spec.requirements
        
        if "api_integration" in requirements.get("context_requirements", []):
            base_requirements.extend(["requests>=2.31.0", "aiohttp>=3.8.0"])
        
        if "data_processing" in requirements.get("context_requirements", []):
            base_requirements.extend(["pandas>=2.0.0", "numpy>=1.24.0"])
        
        if "file_operations" in requirements.get("context_requirements", []):
            base_requirements.append("pathlib2>=2.3.0")
        
        if "analysis_capability" in requirements.get("context_requirements", []):
            base_requirements.extend(["scipy>=1.10.0", "scikit-learn>=1.3.0"])
        
        return base_requirements


class MCPTemplateGenerator:
    """Generates MCP server templates and boilerplate code."""
    
    def __init__(self, model_router):
        self.model_router = model_router
    
    def create_mcp_server(self, spec: MCPSpecification, implementation_code: str) -> str:
        """Create complete MCP server with the generated implementation."""
        
        template = f'''#!/usr/bin/env python3
"""
Generated MCP Server: {spec.name}
Description: {spec.description}
Generated for complexity tier: {spec.model_assignment.get("complexity_tier", "moderate")}
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Simple standalone MCP implementation for compatibility
class SimpleServer:
    def __init__(self, name):
        self.name = name
        self.tools = []
        
    def list_tools(self):
        def decorator(func):
            self.tools.append(func)
            return func
        return decorator
        
    def call_tool(self):
        def decorator(func):
            self.call_tool_handler = func
            return func
        return decorator

# Configure logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'mcp_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MCP Server Configuration
SERVER_NAME = "{spec.name}"
SERVER_VERSION = "{spec.version}"

# Initialize Simple Server
server = SimpleServer(SERVER_NAME)

# === GENERATED IMPLEMENTATION ===
{implementation_code}
# === END GENERATED IMPLEMENTATION ===

@server.list_tools()
async def handle_list_tools():
    """List available tools for this MCP server."""
    return [
        {{
            "name": "{spec.mcp_id}_execute",
            "description": "{spec.description}",
            "inputSchema": {{
                "type": "object",
                "properties": {{
                    "input_data": {{
                        "type": "string",
                        "description": "Input data for processing"
                    }},
                    "parameters": {{
                        "type": "object",
                        "description": "Additional parameters"
                    }}
                }},
                "required": ["input_data"]
            }}
        }}
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict = None):
    """Handle tool execution requests."""
    try:
        if name == "{spec.mcp_id}_execute":
            input_data = arguments.get("input_data", "") if arguments else ""
            parameters = arguments.get("parameters", {{}}) if arguments else {{}}
            
            # Call the generated implementation
            result = await execute_capability(input_data, parameters)
            
            return [{{
                "type": "text",
                "text": json.dumps(result, indent=2)
            }}]
        else:
            raise ValueError(f"Unknown tool: {{name}}")
            
    except Exception as e:
        logger.error(f"Error executing tool {{name}}: {{str(e)}}")
        return [{{
            "type": "text", 
            "text": f"Error: {{str(e)}}"
        }}]

async def execute_main(input_data: str, parameters: dict = None):
    """Main execution function that can be called directly."""
    try:
        return await execute_capability(input_data, parameters or {{}})
    except Exception as e:
        logger.error(f"Execution error: {{str(e)}}")
        return {{"success": False, "error": str(e)}}

async def main():
    """Main entry point for standalone execution."""
    logger.info(f"Starting {{SERVER_NAME}} v{{SERVER_VERSION}} in standalone mode")
    
    # Get parameters from environment if available
    parameters = {{}}
    if "MCP_PARAMETERS" in os.environ:
        try:
            parameters = json.loads(os.environ["MCP_PARAMETERS"])
        except json.JSONDecodeError:
            logger.warning("Failed to parse MCP_PARAMETERS from environment")
    
    # For standalone testing - read from stdin or command line
    if len(sys.argv) > 1:
        input_data = " ".join(sys.argv[1:])
        result = await execute_main(input_data, parameters)
        print(json.dumps(result, indent=2))
    else:
        print(f"{{SERVER_NAME}} MCP Server")
        print(f"Usage: python server.py <input_data>")
        print(f"Description: {{spec.description}}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        return template


class MCPValidator:
    """Validates generated MCP code for security and correctness."""
    
    def __init__(self):
        self.forbidden_imports = [
            "os.system", "subprocess.call", "eval(", "exec(", 
            "__import__", "compile(", "open("
        ]
        self.required_patterns = [
            "async def execute_capability",
            "logger",
            "try:",
            "except"
        ]
    
    def validate_mcp_code(self, code: str) -> Dict[str, Any]:
        """Validate MCP code for security and correctness."""
        validation_results = {
            "is_valid": True,
            "security_issues": [],
            "missing_patterns": [],
            "syntax_valid": False
        }
        
        # Check for forbidden imports/calls
        for forbidden in self.forbidden_imports:
            if forbidden in code:
                validation_results["security_issues"].append(f"Forbidden pattern: {forbidden}")
                validation_results["is_valid"] = False
        
        # Check for required patterns
        for pattern in self.required_patterns:
            if pattern not in code:
                validation_results["missing_patterns"].append(pattern)
                validation_results["is_valid"] = False
        
        # Basic syntax validation
        try:
            compile(code, '<string>', 'exec')
            validation_results["syntax_valid"] = True
        except SyntaxError as e:
            validation_results["syntax_valid"] = False
            validation_results["security_issues"].append(f"Syntax error: {str(e)}")
            validation_results["is_valid"] = False
        
        return validation_results
    
    def sanitize_code(self, code: str) -> str:
        """Basic code sanitization."""
        # Remove potentially dangerous imports
        lines = code.split('\n')
        safe_lines = []
        
        for line in lines:
            # Skip lines with forbidden patterns
            if any(forbidden in line for forbidden in self.forbidden_imports):
                safe_lines.append(f"# REMOVED: {line}")
            else:
                safe_lines.append(line)
        
        return '\n'.join(safe_lines)