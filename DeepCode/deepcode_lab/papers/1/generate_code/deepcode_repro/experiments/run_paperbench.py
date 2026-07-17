import os
import sys
import yaml
import glob
import time
import shutil
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.mcp_tools import MCPToolkit, DockerSandbox
from src.core.document_parser import DocumentSegmenter
from src.core.memory import CodeMem
from src.core.rag_engine import CodeRAG
from src.agents.planning import PlanningSwarm
from src.agents.coding import CodingSwarm
from src.agents.verification import VerificationSwarm

logger = setup_logger("PaperBench")

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    path = os.path.join(project_root, config_path)
    if not os.path.exists(path):
        logger.error(f"Config file not found at {path}")
        raise FileNotFoundError(f"Config file not found at {path}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_evaluation(paper_path: str, output_base_dir: str, config: Dict, run_id: str):
    """Run the full DeepCode pipeline on a single paper."""
    paper_name = Path(paper_path).stem
    repo_name = f"{paper_name}_{run_id}"
    output_dir = os.path.join(output_base_dir, repo_name)
    
    logger.info(f"=== Starting Evaluation for {paper_name} ===")
    logger.info(f"Output Directory: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Tools and Engines
    # 1. Document Parser
    logger.info("Step 1: Parsing Document...")
    segmenter = DocumentSegmenter()
    segments = segmenter.parse_file(paper_path)
    paper_text = "\n\n".join([s.to_text() for s in segments])
    logger.info(f"Parsed {len(segments)} segments from {paper_name}")

    # 2. Planning Phase
    logger.info("Step 2: Planning Phase (Blueprint Synthesis)...")
    planning_swarm = PlanningSwarm(config)
    blueprint = planning_swarm.plan(paper_text)
    
    # Save Blueprint
    blueprint_path = os.path.join(output_dir, "blueprint.json")
    blueprint.save(blueprint_path)
    logger.info(f"Blueprint saved to {blueprint_path}")

    # 3. Coding Phase
    logger.info("Step 3: Coding Phase (Implementation)...")
    # Initialize Memory and RAG
    memory = CodeMem()
    
    # Initialize RAG (Optional based on config, but we'll init it)
    rag_persist_dir = os.path.join(output_dir, ".rag_store")
    rag_engine = CodeRAG(persist_directory=rag_persist_dir)
    # Index the paper segments for RAG
    rag_engine.index_segments(segments, source_id=paper_name)
    
    coding_swarm = CodingSwarm(
        config=config,
        memory=memory,
        rag_engine=rag_engine,
        output_dir=output_dir
    )
    
    coding_swarm.generate_codebase(blueprint)
    
    # Save Memory State
    memory.save(os.path.join(output_dir, "codemem.json"))
    logger.info("Coding Phase Complete.")

    # 4. Verification Phase
    logger.info("Step 4: Verification Phase (Analysis & Sandbox)...")
    
    # Setup Sandbox for Verification
    # We map the output_dir on host to /workspace in container
    sandbox_config = config.get("sandbox", {})
    sandbox = DockerSandbox(
        image=sandbox_config.get("image", "deepcode_sandbox:latest"),
        host_workspace_path=output_dir,
        container_workspace_path="/workspace"
    )
    
    mcp_toolkit = MCPToolkit(sandbox_config={"sandbox": sandbox})
    
    verification_swarm = VerificationSwarm(
        config=config,
        mcp_toolkit=mcp_toolkit,
        output_dir=output_dir
    )
    
    # Run Verification Loop
    verification_swarm.verify_codebase(blueprint)
    logger.info("Verification Phase Complete.")

    # 5. Final Evaluation (Pass/Fail)
    logger.info("Step 5: Final Evaluation...")
    
    # We assume the generated repo has tests (e.g., pytest)
    # The blueprint should have included test files if the paper implied them,
    # or we rely on the verification agent having added them.
    # We'll try to run 'pytest' or 'python -m unittest'
    
    test_passed = False
    with sandbox as sb:
        # Try pytest first
        exit_code, stdout, stderr = sb.execute_command("pytest", timeout=300)
        if exit_code == 0:
            logger.info(f"Tests PASSED for {paper_name} (pytest)")
            test_passed = True
        else:
            logger.warning(f"pytest failed or not found. Output: {stdout[:200]}...")
            # Try unittest
            exit_code, stdout, stderr = sb.execute_command("python -m unittest discover .", timeout=300)
            if exit_code == 0:
                logger.info(f"Tests PASSED for {paper_name} (unittest)")
                test_passed = True
            else:
                logger.error(f"Tests FAILED for {paper_name}")
                logger.error(f"Stdout: {stdout}")
                logger.error(f"Stderr: {stderr}")

    return {
        "paper": paper_name,
        "repo_path": output_dir,
        "passed": test_passed
    }

def main():
    parser = argparse.ArgumentParser(description="Run PaperBench Evaluation")
    parser.add_argument("--papers_dir", type=str, default="data/input_papers", help="Directory containing input papers")
    parser.add_argument("--output_dir", type=str, default="data/output_repos", help="Directory for output repositories")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers to process")
    args = parser.parse_args()

    # Setup paths
    papers_dir = os.path.join(project_root, args.papers_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    
    if not os.path.exists(papers_dir):
        logger.error(f"Papers directory not found: {papers_dir}")
        # Create it for user convenience
        os.makedirs(papers_dir, exist_ok=True)
        logger.info(f"Created {papers_dir}. Please add PDF/MD papers there.")
        return

    # Load Config
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Find papers
    paper_files = glob.glob(os.path.join(papers_dir, "*.pdf")) + glob.glob(os.path.join(papers_dir, "*.md"))
    paper_files.sort()
    
    if args.limit:
        paper_files = paper_files[:args.limit]

    if not paper_files:
        logger.warning("No papers found to process.")
        return

    logger.info(f"Found {len(paper_files)} papers to process.")

    results = []
    run_id = time.strftime("%Y%m%d_%H%M%S")

    for paper_path in paper_files:
        try:
            result = run_evaluation(paper_path, output_dir, config, run_id)
            results.append(result)
        except Exception as e:
            logger.exception(f"Critical failure processing {paper_path}: {e}")
            results.append({
                "paper": Path(paper_path).stem,
                "repo_path": "FAILED",
                "passed": False,
                "error": str(e)
            })

    # Report Summary
    logger.info("=== Evaluation Summary ===")
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\nTotal Papers: {total}")
    print(f"Passed: {passed}")
    print(f"Pass Rate: {pass_rate:.2f}%")
    
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['paper']} -> {r['repo_path']}")

if __name__ == "__main__":
    main()
