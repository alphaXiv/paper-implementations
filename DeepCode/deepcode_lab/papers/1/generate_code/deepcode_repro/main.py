import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import setup_logger, logger
from src.utils.mcp_tools import MCPToolkit, DockerSandbox
from src.core.document_parser import DocumentSegmenter
from src.core.blueprint import Blueprint
from src.core.memory import CodeMem
from src.core.rag_engine import CodeRAG
from src.agents.planning import PlanningSwarm
from src.agents.coding import CodingSwarm
from src.agents.verification import VerificationSwarm

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            sys.exit(1)

def run_planning_phase(paper_path: str, config: Dict[str, Any], output_dir: str) -> Blueprint:
    """
    Phase 1: Planning
    Parse document and generate implementation blueprint.
    """
    logger.info("=== Starting Phase 1: Planning ===")
    
    # 1. Parse Document
    logger.info(f"Parsing document: {paper_path}")
    segmenter = DocumentSegmenter()
    segments = segmenter.parse_file(paper_path)
    
    # Convert segments to full text for the planning agent
    # In a real scenario, we might want to be smarter about this, 
    # but for now we concatenate relevant sections or pass the whole text
    # depending on token limits. The PlanningSwarm expects text.
    full_text = "\n\n".join([s.to_text() for s in segments])
    logger.info(f"Document parsed. Total length: {len(full_text)} characters.")
    
    # 2. Run Planning Swarm
    planning_swarm = PlanningSwarm(config)
    blueprint = planning_swarm.plan(full_text)
    
    # 3. Save Blueprint
    blueprint_path = os.path.join(output_dir, "blueprint.json")
    blueprint.save(blueprint_path)
    logger.info(f"Blueprint saved to {blueprint_path}")
    
    return blueprint

def run_coding_phase(blueprint: Blueprint, config: Dict[str, Any], output_dir: str, rag_engine: Optional[CodeRAG] = None):
    """
    Phase 2: Coding
    Generate code based on blueprint, using CodeMem and CodeRAG.
    """
    logger.info("=== Starting Phase 2: Coding ===")
    
    # 1. Initialize Memory
    memory = CodeMem()
    
    # 2. Initialize Coding Swarm
    coding_swarm = CodingSwarm(
        config=config,
        memory=memory,
        rag_engine=rag_engine,
        output_dir=output_dir
    )
    
    # 3. Generate Codebase
    coding_swarm.generate_codebase(blueprint)
    
    # 4. Save Memory State
    memory_path = os.path.join(output_dir, "codemem_state.json")
    memory.save(memory_path)
    logger.info(f"CodeMem state saved to {memory_path}")

def run_verification_phase(blueprint: Blueprint, config: Dict[str, Any], output_dir: str, sandbox: Optional[DockerSandbox] = None):
    """
    Phase 3: Verification
    Verify generated code using static analysis and sandbox execution.
    """
    logger.info("=== Starting Phase 3: Verification ===")
    
    # 1. Initialize Toolkit with Sandbox
    mcp_toolkit = None
    if sandbox:
        # Create toolkit with existing sandbox configuration
        # The MCPToolkit constructor takes a config dict for the sandbox
        # But here we might want to pass the actual sandbox instance or just the config
        # Looking at MCPToolkit implementation, it creates its own DockerSandbox.
        # We should pass the sandbox config.
        sandbox_config = config.get("sandbox", {})
        # Ensure workspace paths match
        sandbox_config["host_workspace_path"] = output_dir
        mcp_toolkit = MCPToolkit(sandbox_config=sandbox_config)
    
    # 2. Initialize Verification Swarm
    verification_swarm = VerificationSwarm(
        config=config,
        mcp_toolkit=mcp_toolkit,
        output_dir=output_dir
    )
    
    # 3. Verify Codebase
    verification_swarm.verify_codebase(blueprint)

def main():
    parser = argparse.ArgumentParser(description="DeepCode: Open Agentic Coding Framework")
    parser.add_argument("--paper", type=str, help="Path to input research paper (PDF/MD)")
    parser.add_argument("--output", type=str, default="./data/output_repos/generated_project", help="Output directory for generated code")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--blueprint", type=str, help="Path to existing blueprint.json (skip planning)")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification phase")
    parser.add_argument("--use-rag", action="store_true", help="Enable CodeRAG (requires vector store setup)")
    
    args = parser.parse_args()
    
    # Setup Logger
    setup_logger()
    
    # Load Config
    config = load_config(args.config)
    
    # Prepare Output Directory
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize RAG Engine if requested
    rag_engine = None
    if args.use_rag:
        rag_config = config.get("rag", {})
        persist_dir = rag_config.get("persist_directory", "./data/chroma_db")
        collection_name = rag_config.get("collection_name", "deepcode_knowledge")
        rag_engine = CodeRAG(persist_directory=persist_dir, collection_name=collection_name)
        # Note: In a real run, we would index documents here or assume they are indexed.
    
    # --- Phase 1: Planning ---
    blueprint = None
    if args.blueprint:
        logger.info(f"Loading existing blueprint from {args.blueprint}")
        blueprint = Blueprint.load(args.blueprint)
    elif args.paper:
        blueprint = run_planning_phase(args.paper, config, output_dir)
    else:
        logger.error("Either --paper or --blueprint must be provided.")
        sys.exit(1)
        
    # --- Phase 2: Coding ---
    run_coding_phase(blueprint, config, output_dir, rag_engine)
    
    # --- Phase 3: Verification ---
    if not args.skip_verify:
        # Setup Sandbox
        sandbox_config = config.get("sandbox", {})
        # Override host path to point to our output directory
        sandbox_config["host_workspace_path"] = output_dir
        
        # We use the context manager in the swarm or toolkit, but here we might want to 
        # ensure the docker environment is ready. 
        # The VerificationSwarm uses MCPToolkit which manages the sandbox.
        # We just pass the config.
        
        run_verification_phase(blueprint, config, output_dir, sandbox=True) # Pass True to indicate we want sandbox
        
    logger.info("=== DeepCode Execution Complete ===")

if __name__ == "__main__":
    main()
