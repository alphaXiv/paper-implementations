# DeepCode: Open Agentic Coding

A fully autonomous multi-agent framework for high-fidelity document-to-repository synthesis. DeepCode solves the context bottleneck in LLM-based coding via principled information-flow management (CodeMem, CodeRAG, and Blueprinting).

This repository is a reproduction of the framework described in the paper "DeepCode: Open Agentic Coding".

## Core Features

*   **Hierarchical Content Segmentation**: Transforms raw research papers (PDF/Markdown) into structured semantic chunks.
*   **Planning Swarm**: Synthesizes a detailed implementation Blueprint (JSON) using Concept, Algorithm, and Planning agents.
*   **CodeMem**: A graph-based memory system that maintains cross-file consistency without saturating the context window.
*   **CodeRAG**: Retrieval-Augmented Generation engine for injecting external knowledge or specific paper details on demand.
*   **Verification Swarm**: A two-stage verification loop (Static Analysis & Dynamic Sandbox Execution) to ensure functional correctness.
*   **Docker Sandbox**: Secure execution environment for running generated code and tests.

## Prerequisites

*   **OS**: Linux or macOS (required for Docker compatibility)
*   **Python**: >= 3.10
*   **Docker Engine**: Installed and running
*   **API Keys**:
    *   Anthropic (Claude-3.5-Sonnet) OR OpenAI (GPT-4o)
    *   Brave Search (Optional, for external web search)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/deepcode_repro.git
    cd deepcode_repro
    ```

2.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build the Docker Sandbox**:
    The sandbox environment is required for the Verification Agent to run tests safely.
    ```bash
    cd docker
    docker build -t deepcode_sandbox:latest .
    cd ..
    ```

## Configuration

1.  **Edit `config.yaml`**:
    Open `config.yaml` and configure your LLM provider and API keys.

    ```yaml
    llm:
      provider: "anthropic"  # or "openai"
      model: "claude-3-5-sonnet-20240620" # or "gpt-4o"
      api_key: "YOUR_API_KEY_HERE"
      temperature: 0.0
      max_tokens: 4096

    sandbox:
      image: "deepcode_sandbox:latest"
      workspace_mount: "./data/output_repos"
    ```

    *Note: You can also set API keys via environment variables `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.*

## Usage

### 1. Generate a Repository from a Paper

To generate a codebase from a research paper (PDF or Markdown):

```bash
python main.py --paper data/input_papers/attention_is_all_you_need.pdf --output_dir data/output_repos/transformer
```

**Workflow:**
1.  **Parsing**: The paper is chunked into segments.
2.  **Planning**: The Planning Swarm creates a `blueprint.json`.
3.  **Coding**: The Coding Swarm generates files topologically, updating `memory.json`.
4.  **Verification**: The Verification Swarm runs static analysis and executes tests in Docker, applying fixes as needed.

### 2. Run Validation Checks

To verify that the environment, Docker sandbox, and agents are set up correctly:

```bash
python experiments/validate_repro.py
```

### 3. Run PaperBench Evaluation

To run the framework on a dataset of papers (PaperBench) and measure Pass@1 rates:

```bash
python experiments/run_paperbench.py --papers_dir data/input_papers --output_dir data/bench_results --limit 5
```

## Project Structure

```text
deepcode_repro/
├── main.py                         # CLI Entry point
├── config.yaml                     # Configuration
├── docker/                         # Sandbox environment
├── src/
│   ├── core/                       # Core engines
│   │   ├── document_parser.py      # PDF/MD Parsing
│   │   ├── blueprint.py            # Schema definitions
│   │   ├── memory.py               # CodeMem graph
│   │   └── rag_engine.py           # CodeRAG vector store
│   ├── agents/                     # Agent Swarms
│   │   ├── planning.py             # Concept/Algo/Planner agents
│   │   ├── coding.py               # Generator/Summarizer agents
│   │   └── verification.py         # Analysis/Sandbox/Fix agents
│   └── utils/                      # Utilities
│       ├── mcp_tools.py            # Docker & FileSystem tools
│       └── prompts.py              # System prompts
└── experiments/                    # Validation & Benchmarking scripts
```

## Troubleshooting

*   **Docker Connection Error**: Ensure Docker Desktop or the Docker daemon is running. Try `docker ps` to verify.
*   **PDF Parsing Issues**: If `marker-pdf` fails, the system falls back to `pypdf`. Ensure the PDF is text-selectable.
*   **API Rate Limits**: If you encounter rate limits, increase the retry count in `src/agents/base.py` or check your provider's quota.
