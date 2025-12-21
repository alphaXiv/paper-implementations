# DeepCode

This is an implementation of the paper ["DeepCode: Open Agentic Coding"](https://www.alphaxiv.org/abs/2512.07921). DeepCode is a fully autonomous multi-agent framework for high-fidelity document-to-repository synthesis, solving the context bottleneck in LLM-based coding through principled information-flow management.

DeepCode achieves state-of-the-art performance on the PaperBench benchmark, decisively outperforming leading commercial agents such as Cursor and Claude Code, and crucially, surpassing PhD-level human experts from top institutes on key reproduction metrics. Read the paper [here](https://www.alphaxiv.org/abs/2512.07921).

## Quickstart

### One-Line Setup with speedrun_setup.sh

The easiest way to get started is using our `speedrun_setup.sh` script that handles everything:

```bash
# Run the setup wizard
bash speedrun_setup.sh
```

The script automatically:
- Installs `uv` package manager if not present
- Creates Python virtual environment with `uv venv`
- Installs all dependencies
- Configures search servers (Brave or Bocha-MCP)
- Sets up LLM provider (OpenAI, Anthropic, or Google)
- Configures API keys
- Sets up Windows MCP servers (Windows only)

### Manual Setup

If you prefer manual setup or want to install the DeepCode package directly, refer to: [Quick Start Guide](https://github.com/HKUDS/DeepCode?tab=readme-ov-file#-quick-start)

### Launching DeepCode

After setup, you can launch DeepCode in two ways:

**Web Interface:**
```bash
python deepcode.py
```

**CLI Interface:**
```bash
python cli/main_cli.py
```

## Features

### Core Capabilities

- **📄 Document Parsing**: Processes research papers (PDF/Markdown) into structured semantic chunks
- **🧠 Multi-Agent Planning**: Synthesizes detailed implementation blueprints using Concept, Algorithm, and Planning agents
- **💾 CodeMem**: Graph-based memory system maintaining cross-file consistency
- **🔍 CodeRAG**: Retrieval-Augmented Generation for external knowledge injection
- **✅ Verification Swarm**: Two-stage verification (Static Analysis & Dynamic Execution)
- **🐳 Docker Sandbox**: Secure execution environment for running generated code

### Supported LLM Providers

- **OpenAI** (GPT-4o, GPT-4, etc.)
- **Anthropic** (Claude Sonnet, Claude Opus, etc.)
- **Google** (Gemini Pro, etc.)

### Search Integration

- **Brave Search**: Web search for finding similar repositories and code examples
- **Bocha-MCP**: Alternative search server option

## Architecture

DeepCode uses a multi-agent architecture:

- **🎯 Central Orchestration Agent**: Coordinates workflow execution and strategic decisions
- **📝 Requirement Analysis Agent**: Performs deep semantic analysis of user requirements
- **📄 Document Segmentation Agent**: Processes complex technical documents and research papers
- **🏗️ Code Planning Agent**: Executes architectural design and technology stack optimization
- **🔍 Code Reference Mining Agent**: Discovers relevant repositories and frameworks
- **📚 Code Indexing Agent**: Builds comprehensive knowledge graphs of discovered codebases
- **🧬 Code Implementation Agent**: Synthesizes collected information into executable code

All agents communicate via **Model Context Protocol (MCP)** for standardized tool integration.

## Self-Reproduction: DeepCode on DeepCode

As a demonstration of its capabilities, DeepCode was run on its own paper ("DeepCode: Open Agentic Coding"). The framework successfully processed the 23-page research paper and generated a complete reproduction repository. The repo was generated with gemini-3-pro-preview, Brave Search, and with fast mode enabled. 

### Generated Repository Structure

The self-reproduction created a fully functional implementation in `deepcode_lab/papers/1/generate_code/deepcode_repro/`:

**Core Components:**
- `src/core/document_parser.py` - Hierarchical content segmentation
- `src/core/memory.py` - CodeMem graph-based memory system
- `src/core/rag_engine.py` - CodeRAG retrieval engine
- `src/core/blueprint.py` - Blueprint schema definitions

**Agent Implementations:**
- `src/agents/planning.py` - Concept, Algorithm, and Planning agents
- `src/agents/coding.py` - Code generation and summarization agents
- `src/agents/verification.py` - Static analysis and sandbox verification agents
- `src/agents/base.py` - LLM wrapper with MCP tool integration

**Infrastructure:**
- `docker/Dockerfile` - Secure sandbox environment
- `experiments/run_paperbench.py` - PaperBench evaluation script
- `experiments/validate_repro.py` - Validation harness
- Complete `README.md` with installation and usage instructions

## Troubleshooting

- **UV not found**: Make sure `uv` is installed and in your PATH. Restart your terminal after installation.
- **Python version**: DeepCode requires Python 3.13+. Use `uv venv --python=3.13` to create the environment.
- **API key errors**: Verify your API keys are correctly set in `mcp_agent.secrets.yaml` and match your selected LLM provider.
- **Windows MCP servers**: Ensure npm is installed and MCP servers are installed globally before running the Windows configuration step.
- **Virtual environment**: Always activate the virtual environment before running DeepCode: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)

## Source Code

This implementation is based on the original DeepCode framework. Source code: https://github.com/HKUDS/DeepCode
