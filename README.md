# Samus Agent

[An Alita-inspired autonomous agent](https://www.arxiv.org/abs/2505.20286?utm_source=substack&utm_medium=email) using Claude Sonnet 4 for minimal predefinition and maximal self-evolution capabilities through dynamic Model Context Protocol (MCP) generation and management.

**Key Innovation**: Instead of pre-building hundreds of tools, Samus starts with pure reasoning and generates specialized capabilities (MCPs) only when needed. This follows the Alita research principle of minimal predefinition and maximal self-evolution.

## How It Works

### 🧠 **Reasoning-First Approach**
Samus begins every task with direct reasoning using Claude Sonnet 4. It analyzes the problem, breaks it down, and provides comprehensive solutions using its core intelligence.

### 🔧 **Dynamic Capability Generation**
When a task requires specialized tools that don't exist, Samus:
1. **Identifies gaps** in its current capabilities
2. **Generates MCP specifications** describing exactly what's needed  
3. **Creates executable code** using AI to implement the capability
4. **Validates and executes** the new MCP in an isolated process
5. **Stores capabilities** for future reuse and evolution

### 🏗️ **Generated MCP Structure**
Each MCP becomes a complete, runnable server stored in `~/.samus/mcps/`:
```
├── server.py          # Complete MCP server implementation
├── specification.json # Metadata and performance metrics
├── requirements.txt   # Python dependencies
├── start.sh          # Startup script
└── logs/             # Execution logs
```

## User Workflow

### **Simple Tasks** (No MCPs needed)
```bash
samus "What is 2+2?"
samus "Explain how TCP works"
samus "Write a Python function to sort a list"
```
→ Provides direct reasoning-based answers

### **Complex Tasks** (Triggers MCP generation)
```bash
samus "Fetch the current weather in New York using a real API"
samus "Calculate RSI and MACD indicators for Apple stock"
samus "Process this CSV file and generate statistical analysis"
```
→ Generates specialized MCPs, executes them, integrates results

### **Interactive Mode**
```bash
samus --interactive
samus> Analyze the sentiment of recent Tesla tweets
samus> Now visualize the sentiment trends over time
samus> Export the results to a CSV file
```
→ Builds capabilities progressively as conversation evolves

## Installation

### Prerequisites
- Python 3.11+
- [uv package manager](https://github.com/astral-sh/uv)
- OpenRouter API key

### Setup
```bash
# Clone and install
git clone <repository-url>
cd samus-agent-stem
uv pip install -e .

# Set up API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Test installation
samus "Hello, what can you help me with?"
```

## Usage Examples

### **Data Analysis**
```bash
samus "Create a technical analysis MCP for Bitcoin price data with RSI, MACD, and Bollinger Bands"
```

### **API Integration**
```bash
samus "I need to fetch weather data from OpenWeatherMap and create visualizations"
```

### **File Processing**
```bash
samus "Process this Excel file and generate summary statistics with charts"
```

### **Advanced Computation**
```bash
samus "Execute mathematical calculations using symbolic math libraries"
```

## Command-Line Options

```bash
samus [OPTIONS] [PROMPT]

Options:
  -f, --file PATH    Read prompt from file
  -i, --interactive  Start interactive session
  -c, --config PATH  Path to config file
  -v, --verbose      Enable verbose output with reasoning traces
  --version          Show version information
  --help             Show help message
```

## Key Features

### **🎯 Adaptive Intelligence**
- Starts with reasoning, adds tools only when needed
- Each interaction makes the agent more capable
- Learns optimal model assignments for different tasks

### **🔒 Security & Isolation**
- Generated MCPs run in isolated processes
- Code validation prevents malicious patterns
- Sandboxed execution environment

### **📊 Performance Tracking**
- Monitors MCP execution metrics
- Optimizes model selection based on performance
- Tracks cost and efficiency over time

### **🔄 Self-Evolution**
- MCPs improve through usage feedback
- Capability distillation for knowledge transfer
- Automatic optimization of model routing

### **💰 Cost Optimization**
- Uses appropriate models for task complexity
- Lightweight models for simple tasks
- Expert models only when needed

## Advanced Configuration

### **Model Configuration**
Set custom models in configuration:
```python
supervisor_model = "anthropic/claude-sonnet-4"        # Reasoning & coordination
lightweight_model = "anthropic/claude-3.5-haiku"     # Simple tasks
expert_model = "anthropic/claude-opus-4"             # Complex reasoning
```

### **MCP Repository**
MCPs are stored in `~/.samus/mcps/` and can be:
- Shared across agent instances
- Backed up and restored
- Manually inspected and modified

### **Performance Monitoring**
```bash
samus --verbose "Your complex task here"
# Shows:
# - MCPs generated and used
# - Execution time breakdown
# - Model assignments and reasoning
```

## Why Samus?

### **vs Traditional Agents**
- **Traditional**: Pre-built tools, static capabilities, manual integration
- **Samus**: Dynamic generation, self-evolving, reasoning-first approach

### **vs Tool-Heavy Architectures**  
- **Tool-Heavy**: Hundreds of predefined functions, complex orchestration
- **Samus**: Minimal core, generates capabilities on-demand, cleaner architecture

### **vs Static AI Assistants**
- **Static**: Fixed capabilities, can't extend beyond training
- **Samus**: Continuously evolving, adapts to new requirements, learns from usage

## Troubleshooting

### **API Key Issues**
```bash
# Check if API key is set
echo $OPENROUTER_API_KEY

# Verify in .env file
cat .env
```

### **MCP Generation Failures**
```bash
# Check MCP directory
ls ~/.samus/mcps/

# View logs
cat ~/.samus/mcps/*/logs/mcp_server.log
```

### **Performance Issues**
- Use `--verbose` to see execution breakdown
- Check model assignments in MCP specifications
- Monitor API usage and rate limits

## Contributing

This implementation follows the Samus research paper principles:
1. **Minimal predefinition** - Start simple, evolve as needed
2. **Maximal self-evolution** - Capabilities emerge from usage
3. **Dynamic capability acquisition** - Generate tools on-demand
4. **Performance-based optimization** - Learn from execution metrics

## License

MIT License - See LICENSE file for details