import os
import logging
import docker
import requests
import shutil
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import json
import time

# Try to import logger, fallback if not available
try:
    from src.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger("DeepCode")
    logging.basicConfig(level=logging.INFO)

class DockerSandbox:
    """
    Manages a Docker container for secure execution of generated code.
    Ensures that code runs in an isolated environment with access only to the workspace.
    """
    def __init__(self, 
                 image: str = "deepcode_sandbox:latest", 
                 host_workspace_path: str = "./data/output_repos",
                 container_workspace_path: str = "/workspace",
                 auto_remove: bool = True):
        """
        Initialize the Docker Sandbox.
        
        Args:
            image: Docker image to use.
            host_workspace_path: Path on host to mount.
            container_workspace_path: Path inside container where host path is mounted.
            auto_remove: Whether to remove the container when stopped.
        """
        self.image = image
        self.host_workspace_path = os.path.abspath(host_workspace_path)
        self.container_workspace_path = container_workspace_path
        self.auto_remove = auto_remove
        self.client = docker.from_env()
        self.container = None
        
        # Ensure host workspace exists
        os.makedirs(self.host_workspace_path, exist_ok=True)

    def start(self):
        """Start the sandbox container."""
        try:
            # Check if image exists, if not try to pull or build (simplified to check)
            try:
                self.client.images.get(self.image)
            except docker.errors.ImageNotFound:
                logger.warning(f"Image {self.image} not found. Attempting to build from ./docker context if available or pull.")
                # Logic to build or pull could go here. For now, we assume it exists or will be built by setup.
                # In a real scenario, we might trigger a build here.
                pass

            logger.info(f"Starting sandbox container with image {self.image}...")
            self.container = self.client.containers.run(
                self.image,
                command="tail -f /dev/null",  # Keep alive
                detach=True,
                volumes={self.host_workspace_path: {'bind': self.container_workspace_path, 'mode': 'rw'}},
                working_dir=self.container_workspace_path,
                auto_remove=self.auto_remove,
                network_mode="bridge" # Allow some network for pip install if needed, or 'none' for strict isolation
            )
            logger.info(f"Sandbox container started: {self.container.id[:10]}")
        except Exception as e:
            logger.error(f"Failed to start sandbox: {e}")
            raise

    def stop(self):
        """Stop the sandbox container."""
        if self.container:
            try:
                logger.info("Stopping sandbox container...")
                self.container.stop()
                self.container = None
            except Exception as e:
                logger.error(f"Error stopping sandbox: {e}")

    def execute_command(self, command: str, timeout: int = 30) -> Tuple[int, str, str]:
        """
        Execute a command inside the sandbox.
        
        Args:
            command: Shell command to execute.
            timeout: Execution timeout in seconds.
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        if not self.container:
            raise RuntimeError("Sandbox not started. Call start() first.")
        
        logger.debug(f"Executing in sandbox: {command}")
        try:
            # docker exec_run doesn't support timeout natively in the same way as subprocess
            # but we can implement it if needed. For now, simple exec.
            exec_result = self.container.exec_run(
                cmd=f"bash -c '{command}'",
                workdir=self.container_workspace_path
            )
            
            exit_code = exec_result.exit_code
            output = exec_result.output.decode('utf-8', errors='replace')
            
            # Split stdout/stderr roughly (Docker API combines them usually unless tty=False and stream=True)
            # For simple usage, we return output as stdout and empty stderr unless we parse it.
            # To get separate streams, we'd need to use socket attachment. 
            # For this reproduction, combined output is often sufficient or we assume stdout.
            
            return exit_code, output, "" 
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return -1, "", str(e)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class FileSystemTool:
    """
    Provides safe filesystem access restricted to a specific root directory.
    Used by agents to read/write code files on the host (which are then mounted to sandbox).
    """
    def __init__(self, root_path: str = "./data/output_repos"):
        self.root_path = os.path.abspath(root_path)
        os.makedirs(self.root_path, exist_ok=True)

    def _validate_path(self, path: str) -> Path:
        """Ensure path is within root_path."""
        full_path = (Path(self.root_path) / path).resolve()
        if not str(full_path).startswith(self.root_path):
            raise ValueError(f"Access denied: Path {path} is outside sandbox root {self.root_path}")
        return full_path

    def list_files(self, path: str = ".") -> List[str]:
        """List files in a directory relative to root."""
        target_path = self._validate_path(path)
        if not target_path.exists():
            return []
        
        files = []
        for p in target_path.rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(self.root_path)))
        return files

    def read_file(self, path: str) -> str:
        """Read content of a file."""
        target_path = self._validate_path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            return target_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file. Creates directories if needed."""
        target_path = self._validate_path(path)
        
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding='utf-8')
            logger.info(f"Written file: {path}")
            return f"Successfully wrote to {path}"
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            raise


class BraveSearchTool:
    """
    Wrapper for Brave Search API to allow agents to retrieve external information.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Execute a search query.
        """
        if not self.api_key:
            logger.warning("Brave Search API key not provided. Returning empty results.")
            return []

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        try:
            response = requests.get(
                self.base_url,
                params={"q": query, "count": count},
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "web" in data and "results" in data["web"]:
                for item in data["web"]["results"]:
                    results.append({
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "description": item.get("description")
                    })
            return results
        except Exception as e:
            logger.error(f"Brave Search failed: {e}")
            return []


class MCPToolkit:
    """
    Aggregates all tools into a single interface for Agents.
    """
    def __init__(self, sandbox_config: Dict[str, Any] = None):
        self.config = sandbox_config or {}
        
        # Initialize tools
        self.fs = FileSystemTool(root_path=self.config.get("host_workspace", "./data/output_repos"))
        self.sandbox = DockerSandbox(
            image=self.config.get("image", "deepcode_sandbox:latest"),
            host_workspace_path=self.config.get("host_workspace", "./data/output_repos"),
            container_workspace_path=self.config.get("container_workspace", "/workspace")
        )
        self.search = BraveSearchTool(api_key=self.config.get("brave_api_key"))

    def get_tools_description(self) -> str:
        """Returns a text description of available tools for the LLM system prompt."""
        return """
        Available Tools:
        1. read_file(path): Read content of a file.
        2. write_file(path, content): Write content to a file.
        3. list_files(path): List all files in directory.
        4. execute_command(command): Execute a shell command in the sandbox environment.
        5. search_web(query): Search the web for information.
        """
