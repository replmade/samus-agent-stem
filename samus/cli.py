"""CLI interface for Samus agent."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from .agent import SamusAgent
from .config import Config


@click.group()
@click.option("--config", "-c", type=click.Path(), help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.version_option()
@click.pass_context
def main(ctx, config: Optional[str], verbose: bool) -> None:
    """Samus Agent - Autonomous agent with dynamic MCP capabilities."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose


@main.command()
@click.argument("prompt", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read prompt from file")
@click.option("--interactive", "-i", is_flag=True, help="Start interactive session")
@click.pass_context
def chat(
    ctx,
    prompt: Optional[str],
    file: Optional[str],
    interactive: bool,
) -> None:
    """Chat with the Samus agent."""
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config_obj = Config.load(ctx.obj['config_path'])
    verbose = ctx.obj['verbose']
    
    # Initialize agent
    agent = SamusAgent(config_obj)
    
    try:
        if interactive:
            asyncio.run(_run_interactive_session(agent, verbose))
        elif file:
            prompt_text = Path(file).read_text().strip()
            asyncio.run(_run_single_prompt(agent, prompt_text, verbose))
        elif prompt:
            asyncio.run(_run_single_prompt(agent, prompt, verbose))
        else:
            click.echo("Error: Must provide a prompt, file, or use interactive mode")
            sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


async def _run_single_prompt(agent: SamusAgent, prompt: str, verbose: bool) -> None:
    """Run a single prompt through the agent."""
    if verbose:
        click.echo(f"Processing: {prompt[:100]}...")
    
    try:
        result = await agent.process(prompt)
        click.echo(result.content)
        
        if verbose:
            click.echo(f"\nUsed MCPs: {', '.join(result.mcps_used)}")
            click.echo(f"Execution time: {result.execution_time:.2f}s")
            if result.reasoning_trace:
                click.echo(f"Reasoning trace: {', '.join(result.reasoning_trace)}")
    finally:
        await agent.shutdown()


async def _run_interactive_session(agent: SamusAgent, verbose: bool) -> None:
    """Run an interactive session with the agent."""
    click.echo("Samus Agent Interactive Session")
    click.echo("Type 'exit' or 'quit' to end the session\n")
    
    try:
        while True:
            try:
                prompt = click.prompt("samus>", type=str)
                
                if prompt.lower() in ["exit", "quit"]:
                    break
                    
                if not prompt.strip():
                    continue
                    
                result = await agent.process(prompt)
                click.echo(f"\n{result.content}\n")
                
                if verbose:
                    click.echo(f"Used MCPs: {', '.join(result.mcps_used)}")
                    click.echo(f"Execution time: {result.execution_time:.2f}s")
                    if result.reasoning_trace:
                        click.echo(f"Reasoning trace: {', '.join(result.reasoning_trace)}\n")
                    
            except (EOFError, KeyboardInterrupt):
                break
    finally:
        await agent.shutdown()


@main.command()
@click.option("--target-model", "-t", required=True, help="Target model for distillation")
@click.option("--performance-threshold", "-p", default=0.7, help="Minimum performance threshold")
@click.option("--max-mcps", "-m", type=int, help="Maximum number of MCPs to distill")
@click.option("--export-path", "-e", type=click.Path(), help="Path to export distilled MCPs")
@click.pass_context
def distill(
    ctx,
    target_model: str,
    performance_threshold: float,
    max_mcps: Optional[int],
    export_path: Optional[str],
) -> None:
    """Distill current agent capabilities for a target model."""
    
    async def run_distillation():
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config_obj = Config.load(ctx.obj['config_path'])
        verbose = ctx.obj['verbose']
        
        try:
            # Initialize source agent
            source_agent = SamusAgent(config_obj)
            
            # Get source MCPs
            source_mcps = source_agent.get_all_mcps()
            
            if not source_mcps:
                click.echo("No MCPs found in current agent. Generate some capabilities first.")
                return
            
            if verbose:
                click.echo(f"Found {len(source_mcps)} MCPs in source agent")
                source_stats = source_agent.get_agent_statistics()
                click.echo(f"Source agent statistics: {source_stats}")
            
            # Import distillation classes
            from .distilled_agent import DistilledAgent
            
            # Create distilled agent
            click.echo(f"Creating distilled agent for model: {target_model}")
            distilled_agent = DistilledAgent.create_from_distillation(
                config_obj,
                source_mcps,
                target_model,
                source_agent_id="current_agent",
                performance_threshold=performance_threshold,
                max_mcps=max_mcps
            )
            
            # Display distillation results
            if distilled_agent.distillation_info:
                metrics = distilled_agent.distillation_info.distillation_metrics
                click.echo(f"\nDistillation Results:")
                click.echo(f"  Inherited MCPs: {distilled_agent.distillation_info.inherited_mcps_count}")
                click.echo(f"  Transfer Success Rate: {metrics.transfer_success_rate:.2f}")
                click.echo(f"  Performance Retention: {metrics.performance_retention:.2f}")
                click.echo(f"  Adaptation Quality: {metrics.adaptation_quality:.2f}")
                click.echo(f"  Transfer Time: {metrics.transfer_time:.2f}s")
            
            # Export if requested
            if export_path:
                from pathlib import Path
                from .distillation import AgentDistillationEngine
                from .models import ModelRouter
                
                model_router = ModelRouter(config_obj)
                distillation_engine = AgentDistillationEngine(config_obj, model_router)
                
                success = distillation_engine.export_mcps(
                    distilled_agent.inherited_mcps,
                    Path(export_path)
                )
                
                if success:
                    click.echo(f"Exported distilled MCPs to: {export_path}")
                else:
                    click.echo("Failed to export MCPs")
            
            # Display capability summary
            if verbose:
                summary = distilled_agent.get_capability_summary()
                click.echo(f"\nDistilled Agent Capabilities:")
                click.echo(f"  Inherited: {summary['inherited_capabilities']['count']} MCPs")
                click.echo(f"  Domains: {summary['inherited_capabilities']['domains']}")
                click.echo(f"  Avg Performance: {summary['inherited_capabilities']['avg_performance']:.2f}")
            
            await source_agent.shutdown()
            await distilled_agent.shutdown()
            
        except Exception as e:
            click.echo(f"Distillation failed: {str(e)}", err=True)
    
    asyncio.run(run_distillation())


@main.command()
@click.pass_context
def stats(ctx) -> None:
    """Show agent statistics and MCP information."""
    
    async def show_stats():
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config_obj = Config.load(ctx.obj['config_path'])
        
        try:
            # Initialize agent
            agent = SamusAgent(config_obj)
            
            # Get statistics
            stats = agent.get_agent_statistics()
            
            click.echo("Agent Statistics:")
            click.echo(f"  Total MCPs: {stats['total_mcps']}")
            click.echo(f"  High Performance MCPs: {stats['high_performance_mcps']}")
            click.echo(f"  Average Success Rate: {stats['avg_success_rate']:.2f}")
            click.echo(f"  Total Executions: {stats['total_executions']}")
            
            if stats['domains']:
                click.echo("  Domains:")
                for domain, count in stats['domains'].items():
                    click.echo(f"    {domain}: {count} MCPs")
            
            # List individual MCPs
            all_mcps = agent.get_all_mcps()
            if all_mcps:
                click.echo(f"\nIndividual MCPs:")
                for mcp in all_mcps:
                    metrics = mcp.performance_metrics
                    success_rate = metrics.get("success_rate", 0.0)
                    executions = metrics.get("execution_count", 0)
                    click.echo(f"  {mcp.name}: {success_rate:.2f} success rate, {executions} executions")
            
            await agent.shutdown()
            
        except Exception as e:
            click.echo(f"Failed to get statistics: {str(e)}", err=True)
    
    asyncio.run(show_stats())


@main.command()
@click.argument("import_path", type=click.Path(exists=True))
@click.option("--target-model", "-t", help="Target model for imported MCPs")
@click.pass_context
def import_mcps(ctx, import_path: str, target_model: Optional[str]) -> None:
    """Import MCPs from exported file."""
    
    async def run_import():
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config_obj = Config.load(ctx.obj['config_path'])
        
        try:
            from pathlib import Path
            from .distillation import AgentDistillationEngine
            from .models import ModelRouter
            
            model_router = ModelRouter(config_obj)
            distillation_engine = AgentDistillationEngine(config_obj, model_router)
            
            # Import MCPs
            imported_mcps = distillation_engine.import_mcps(Path(import_path))
            
            if not imported_mcps:
                click.echo("No MCPs found in import file")
                return
            
            click.echo(f"Imported {len(imported_mcps)} MCPs")
            
            # If target model specified, create distilled agent
            if target_model:
                from .distilled_agent import DistilledAgent
                
                distilled_agent = DistilledAgent.create_from_distillation(
                    config_obj,
                    imported_mcps,
                    target_model,
                    source_agent_id="imported"
                )
                
                click.echo(f"Created distilled agent for {target_model}")
                
                summary = distilled_agent.get_capability_summary()
                click.echo(f"Inherited {summary['inherited_capabilities']['count']} capabilities")
                
                await distilled_agent.shutdown()
            else:
                # Just store the imported MCPs
                from .mcp import MCPRepository
                repository = MCPRepository(config_obj)
                
                for mcp in imported_mcps:
                    repository.store_mcp(mcp)
                
                click.echo(f"Stored {len(imported_mcps)} MCPs in repository")
            
        except Exception as e:
            click.echo(f"Import failed: {str(e)}", err=True)
    
    asyncio.run(run_import())


if __name__ == "__main__":
    main()