import os
import sys
import logging
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.mcp_tools import DockerSandbox
from src.core.document_parser import DocumentSegmenter
from src.core.blueprint import Blueprint
from src.core.memory import CodeMem
from src.agents.planning import PlanningSwarm
from src.agents.coding import CodingSwarm
from src.agents.verification import VerificationSwarm

logger = setup_logger("ValidateRepro")

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def check_docker():
    logger.info("Checking Docker Sandbox...")
    try:
        # We assume the image is built or will be built. 
        # If not built, this might take time or fail if docker is not running.
        # We use a short timeout for the command.
        with DockerSandbox() as sandbox:
            exit_code, stdout, stderr = sandbox.execute_command("echo 'Hello from Sandbox'", timeout=10)
            if exit_code == 0 and "Hello from Sandbox" in stdout:
                logger.info("✅ Docker Sandbox is working.")
            else:
                logger.error(f"❌ Docker Sandbox failed. Output: {stdout}, Error: {stderr}")
    except Exception as e:
        logger.error(f"❌ Docker Sandbox check failed: {e}")
        logger.warning("Ensure Docker is running and the image 'deepcode_sandbox:latest' is built (or can be built).")

def check_document_parser():
    logger.info("Checking Document Parser...")
    try:
        segmenter = DocumentSegmenter()
        text = "# Header 1\nContent 1\n## Header 2\nContent 2"
        segments = segmenter.parse_markdown(text)
        if len(segments) == 2:
            logger.info("✅ Document Parser (Markdown) is working.")
        else:
            logger.error(f"❌ Document Parser failed. Expected 2 segments, got {len(segments)}")
    except Exception as e:
        logger.error(f"❌ Document Parser check failed: {e}")

def check_agents_init(config):
    logger.info("Checking Agent Initialization...")
    try:
        # Planning
        PlanningSwarm(config)
        logger.info("✅ PlanningSwarm initialized.")
        
        # Coding
        mem = CodeMem()
        # We pass None for rag_engine and a dummy output dir
        CodingSwarm(config, mem, None, "data/output_repos")
        logger.info("✅ CodingSwarm initialized.")
        
        # Verification
        VerificationSwarm(config, None, "data/output_repos")
        logger.info("✅ VerificationSwarm initialized.")
        
    except Exception as e:
        logger.error(f"❌ Agent initialization failed: {e}")

def main():
    logger.info("Starting DeepCode Reproduction Validation...")
    
    config = load_config()
    if not config:
        return

    # Check API Keys
    llm_config = config.get("llm", {})
    if not llm_config.get("api_key"):
        logger.warning("⚠️ LLM API Key not found in config. Agents will fail if run.")
    else:
        logger.info(f"✅ API Key present for provider: {llm_config.get('provider')}")
    
    check_document_parser()
    check_agents_init(config)
    
    # Docker check might require the daemon, so we wrap it carefully
    check_docker()
    
    logger.info("Validation Complete.")

if __name__ == "__main__":
    main()
