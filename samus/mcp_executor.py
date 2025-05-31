"""MCP execution engine with process isolation and communication."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .mcp import MCPSpecification


class MCPExecutor:
    """Executes MCP servers in isolated processes with communication protocol."""
    
    def __init__(self, config: Config):
        self.config = config
        self.active_mcps: Dict[str, MCPProcess] = {}
        self.logger = logging.getLogger(__name__)
    
    async def execute_mcp(
        self, 
        spec: MCPSpecification, 
        input_data: str, 
        parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute an MCP with the given input data using direct execution."""
        
        start_time = time.time()
        
        try:
            # Execute using simplified direct approach
            result = await self._execute_direct(spec, input_data, parameters or {})
            
            execution_time = time.time() - start_time
            
            # Record performance metrics
            await self._record_performance(spec, execution_time, True, result)
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "mcp_id": spec.mcp_id
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"MCP execution failed for {spec.mcp_id}: {str(e)}")
            
            # Record failure metrics
            await self._record_performance(spec, execution_time, False, str(e))
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "mcp_id": spec.mcp_id
            }
    
    async def _execute_direct(self, spec: MCPSpecification, input_data: str, parameters: Dict) -> Dict[str, Any]:
        """Execute MCP using direct Python subprocess execution."""
        
        # Get the MCP server file path
        mcp_path = Path(self.config.mcp_repository_path) / spec.mcp_id / "server.py"
        
        if not mcp_path.exists():
            raise RuntimeError(f"MCP server file not found: {mcp_path}")
        
        # Prepare execution arguments
        cmd = [
            sys.executable,
            str(mcp_path),
            input_data
        ]
        
        # Set environment variables for parameters
        env = os.environ.copy()
        env["MCP_PARAMETERS"] = json.dumps(parameters)
        
        try:
            # Execute the MCP server
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=mcp_path.parent
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=30.0  # 30 second timeout
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"MCP execution failed: {error_msg}")
            
            # Parse the result
            output = stdout.decode().strip()
            if output:
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return {"success": True, "data": output}
            else:
                return {"success": True, "data": "No output"}
                
        except asyncio.TimeoutError:
            raise RuntimeError("MCP execution timed out")
        except Exception as e:
            raise RuntimeError(f"MCP execution error: {str(e)}")
    
    async def _get_or_start_mcp(self, spec: MCPSpecification) -> "MCPProcess":
        """Get existing MCP process or start a new one."""
        
        if spec.mcp_id in self.active_mcps:
            mcp_process = self.active_mcps[spec.mcp_id]
            if mcp_process.is_healthy():
                return mcp_process
            else:
                # Process is unhealthy, clean up and restart
                await mcp_process.terminate()
                del self.active_mcps[spec.mcp_id]
        
        # Start new MCP process
        mcp_process = MCPProcess(spec, self.config)
        await mcp_process.start()
        
        self.active_mcps[spec.mcp_id] = mcp_process
        return mcp_process
    
    async def _record_performance(
        self, 
        spec: MCPSpecification, 
        execution_time: float, 
        success: bool, 
        result: Any
    ) -> None:
        """Record performance metrics for the MCP execution."""
        
        # Update in-memory performance metrics
        metrics = spec.performance_metrics
        
        # Calculate running averages
        current_count = metrics.get("execution_count", 0)
        current_avg_time = metrics.get("execution_time", 0.0)
        current_success_rate = metrics.get("success_rate", 0.0)
        
        new_count = current_count + 1
        new_avg_time = ((current_avg_time * current_count) + execution_time) / new_count
        new_success_rate = ((current_success_rate * current_count) + (1.0 if success else 0.0)) / new_count
        
        # Update metrics
        spec.performance_metrics.update({
            "execution_count": new_count,
            "execution_time": new_avg_time,
            "success_rate": new_success_rate,
            "last_execution": time.time(),
            "last_success": success
        })
        
        # Save updated specification
        await self._save_updated_spec(spec)
        
        # Also update in the main repository
        await self._update_repository_metrics(spec)
    
    async def _save_updated_spec(self, spec: MCPSpecification) -> None:
        """Save updated specification with performance metrics."""
        spec_file = Path(self.config.mcp_repository_path) / spec.mcp_id / "specification.json"
        
        if spec_file.exists():
            spec_data = {
                "mcp_id": spec.mcp_id,
                "version": spec.version,
                "name": spec.name,
                "description": spec.description,
                "model_assignment": spec.model_assignment,
                "requirements": spec.requirements,
                "implementation": spec.implementation,
                "performance_metrics": spec.performance_metrics,
                "evolution_history": spec.evolution_history
            }
            
            with open(spec_file, 'w') as f:
                json.dump(spec_data, f, indent=2)
    
    async def _update_repository_metrics(self, spec: MCPSpecification) -> None:
        """Update metrics in the main MCP repository."""
        try:
            from .mcp import MCPRepository
            repository = MCPRepository(self.config)
            repository.store_mcp(spec)  # This will update the existing MCP
        except Exception as e:
            self.logger.warning(f"Failed to update repository metrics: {str(e)}")
    
    async def shutdown_all_mcps(self) -> None:
        """Shutdown all active MCP processes."""
        for mcp_id, mcp_process in self.active_mcps.items():
            try:
                await mcp_process.terminate()
            except Exception as e:
                self.logger.error(f"Error shutting down MCP {mcp_id}: {str(e)}")
        
        self.active_mcps.clear()
    
    async def get_mcp_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active MCPs."""
        status = {}
        
        for mcp_id, mcp_process in self.active_mcps.items():
            status[mcp_id] = {
                "is_healthy": mcp_process.is_healthy(),
                "uptime": mcp_process.get_uptime(),
                "last_activity": mcp_process.last_activity,
                "process_id": mcp_process.process_id
            }
        
        return status


class MCPProcess:
    """Manages a single MCP server process with communication."""
    
    def __init__(self, spec: MCPSpecification, config: Config):
        self.spec = spec
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.last_activity: Optional[float] = None
        self.process_id: Optional[int] = None
        self.logger = logging.getLogger(f"mcp.{spec.mcp_id}")
    
    async def start(self) -> None:
        """Start the MCP server process."""
        
        mcp_path = Path(self.config.mcp_repository_path) / self.spec.mcp_id
        server_file = mcp_path / "server.py"
        
        if not server_file.exists():
            raise RuntimeError(f"MCP server file not found: {server_file}")
        
        # Prepare environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(mcp_path)
        
        try:
            # Start the MCP server process
            self.process = subprocess.Popen(
                [sys.executable, str(server_file)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(mcp_path),
                env=env,
                text=True,
                bufsize=0
            )
            
            self.process_id = self.process.pid
            self.start_time = time.time()
            self.last_activity = time.time()
            
            # Wait for process to initialize
            await asyncio.sleep(0.5)
            
            if self.process.poll() is not None:
                # Process died immediately
                stderr = self.process.stderr.read() if self.process.stderr else "Unknown error"
                raise RuntimeError(f"MCP process failed to start: {stderr}")
            
            self.logger.info(f"Started MCP process {self.process_id} for {self.spec.mcp_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP process: {str(e)}")
            raise
    
    async def execute(self, input_data: str, parameters: Dict[str, Any]) -> Any:
        """Execute a capability request on the MCP server."""
        
        if not self.is_healthy():
            raise RuntimeError("MCP process is not healthy")
        
        # Prepare MCP tool call request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": f"{self.spec.mcp_id}_execute",
                "arguments": {
                    "input_data": input_data,
                    "parameters": parameters
                }
            }
        }
        
        try:
            # Send request to MCP process
            request_json = json.dumps(request) + '\n'
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Read response (with timeout)
            response_line = await asyncio.wait_for(
                asyncio.create_task(self._read_response()),
                timeout=self.config.request_timeout
            )
            
            response = json.loads(response_line)
            self.last_activity = time.time()
            
            # Handle MCP response
            if "error" in response:
                raise RuntimeError(f"MCP error: {response['error']}")
            
            # Extract result from MCP response
            result = response.get("result", {})
            if isinstance(result, list) and len(result) > 0:
                # Extract text content from MCP response
                content = result[0].get("text", "")
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"MCP execution timeout for {self.spec.mcp_id}")
            raise RuntimeError("MCP execution timeout")
        except Exception as e:
            self.logger.error(f"MCP execution error: {str(e)}")
            raise
    
    async def _read_response(self) -> str:
        """Read response from MCP process."""
        return self.process.stdout.readline()
    
    def is_healthy(self) -> bool:
        """Check if the MCP process is healthy."""
        if not self.process:
            return False
        
        # Check if process is still running
        if self.process.poll() is not None:
            return False
        
        # Check if process is responsive (within last 5 minutes)
        if self.last_activity and time.time() - self.last_activity > 300:
            return False
        
        return True
    
    def get_uptime(self) -> Optional[float]:
        """Get uptime of the MCP process in seconds."""
        if not self.start_time:
            return None
        return time.time() - self.start_time
    
    async def terminate(self) -> None:
        """Terminate the MCP process."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)
                
                # Force kill if still running
                if self.process.poll() is None:
                    self.process.kill()
                    await asyncio.sleep(0.5)
                
                self.logger.info(f"Terminated MCP process {self.process_id}")
                
            except Exception as e:
                self.logger.error(f"Error terminating MCP process: {str(e)}")
            finally:
                self.process = None
                self.process_id = None


class MCPCommunicationProtocol:
    """Handles communication protocol between agent and MCPs."""
    
    @staticmethod
    def create_tool_call_request(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized MCP tool call request."""
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
    
    @staticmethod
    def parse_tool_response(response: Dict[str, Any]) -> Any:
        """Parse MCP tool call response."""
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        
        result = response.get("result", {})
        
        # Handle different response formats
        if isinstance(result, list):
            # Extract text content from MCP response
            if len(result) > 0 and "text" in result[0]:
                content = result[0]["text"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
        
        return result