"""CLI interface for Samus agent."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from .agent import SamusAgent
from .config import Config


@click.command()
@click.argument("prompt", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read prompt from file")
@click.option("--interactive", "-i", is_flag=True, help="Start interactive session")
@click.option("--config", "-c", type=click.Path(), help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.version_option()
def main(
    prompt: Optional[str],
    file: Optional[str],
    interactive: bool,
    config: Optional[str],
    verbose: bool,
) -> None:
    """Samus Agent - Autonomous agent with dynamic MCP capabilities."""
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config_obj = Config.load(config)
    
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


if __name__ == "__main__":
    main()