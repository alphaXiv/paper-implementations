import os
import re
from typing import Dict, Any, Optional, List, Tuple
from deepcode_repro.src.agents.base import BaseAgent
from deepcode_repro.src.core.blueprint import Blueprint, ComponentSpecification
from deepcode_repro.src.utils.prompts import AgentPrompts
from deepcode_repro.src.utils.logger import logger
from deepcode_repro.src.utils.mcp_tools import MCPToolkit

class AnalysisAgent(BaseAgent):
    """
    Agent responsible for static analysis of generated code against the blueprint specification.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="AnalysisAgent", config=config)
        self.system_prompt = AgentPrompts.ANALYSIS_AGENT_SYSTEM

    def run(self, code: str, spec: ComponentSpecification) -> Optional[str]:
        """
        Analyzes the code and returns a report if issues are found, or None if clean.
        """
        spec_str = f"Purpose: {spec.core_purpose}\nInterface: {spec.public_interface}"
        task = AgentPrompts.get_analysis_task(code, spec_str)
        
        response = self.call_llm(self.system_prompt, task)
        
        # Heuristic: If response contains "NO ISSUES FOUND" (case insensitive), return None
        if "NO ISSUES FOUND" in response.upper():
            return None
            
        return response

class ModificationAgent(BaseAgent):
    """
    Agent responsible for applying fixes to code based on feedback (static analysis or runtime errors).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="ModificationAgent", config=config)
        self.system_prompt = AgentPrompts.MODIFICATION_AGENT_SYSTEM

    def run(self, code: str, feedback: str) -> str:
        """
        Generates corrected code based on the provided feedback.
        """
        task = AgentPrompts.get_modification_task(code, feedback)
        response = self.call_llm(self.system_prompt, task)
        
        # Extract code block
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Fallback: if no block, assume full response is code (risky but necessary)
        # or try to find just ``` ... ```
        code_match_generic = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if code_match_generic:
            return code_match_generic.group(1)
            
        return response

class SandboxAgent(BaseAgent):
    """
    Agent responsible for analyzing runtime traces and diagnosing errors.
    Note: The actual execution happens via the VerificationSwarm using MCP tools.
    This agent acts as the 'Brain' analyzing the 'Trace'.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="SandboxAgent", config=config)
        self.system_prompt = AgentPrompts.SANDBOX_AGENT_SYSTEM

    def analyze_trace(self, code: str, trace: str) -> str:
        """
        Analyzes the execution trace and returns fix instructions.
        """
        # We reuse the modification task prompt structure or create a specific one.
        # The prompt template for modification takes 'feedback', which can be the trace analysis.
        # But here we want to produce the analysis/instructions for the ModificationAgent.
        
        # Let's construct a specific prompt for diagnosis
        task = f"""
        Analyze the following execution trace for the provided code.
        Identify the root cause of the error and provide specific instructions to fix it.
        
        CODE:
        ```python
        {code}
        ```
        
        EXECUTION TRACE:
        {trace}
        
        OUTPUT:
        Provide a concise set of fix instructions.
        """
        
        response = self.call_llm(self.system_prompt, task)
        return response

class VerificationSwarm:
    """
    Orchestrates the verification loop: Static Analysis -> Dynamic Execution -> Self-Correction.
    """
    def __init__(self, config: Dict[str, Any], mcp_toolkit: Optional[MCPToolkit], output_dir: str):
        self.config = config
        self.mcp_toolkit = mcp_toolkit
        self.output_dir = output_dir
        
        self.analysis_agent = AnalysisAgent(config)
        self.modification_agent = ModificationAgent(config)
        self.sandbox_agent = SandboxAgent(config)
        
        self.max_retries = 3

    def verify_file_static(self, file_path: str, spec: ComponentSpecification) -> bool:
        """
        Performs static analysis and correction loop.
        Returns True if verified (or fixed), False if issues persist.
        """
        full_path = os.path.join(self.output_dir, file_path)
        if not os.path.exists(full_path):
            logger.error(f"File not found for verification: {full_path}")
            return False

        logger.info(f"Starting Static Verification for {file_path}")
        
        for i in range(self.max_retries + 1):
            # Read current code
            try:
                with open(full_path, 'r') as f:
                    code = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {full_path}: {e}")
                return False

            # Analyze
            report = self.analysis_agent.run(code, spec)
            
            if report is None:
                logger.info(f"Static Analysis passed for {file_path}")
                return True
            
            logger.warning(f"Static Analysis issues found in {file_path} (Attempt {i+1}/{self.max_retries + 1})")
            logger.debug(f"Report: {report}")
            
            if i == self.max_retries:
                logger.error(f"Max retries reached for static verification of {file_path}")
                return False
            
            # Fix
            logger.info(f"Applying fixes to {file_path}...")
            new_code = self.modification_agent.run(code, report)
            
            # Save fixed code
            try:
                with open(full_path, 'w') as f:
                    f.write(new_code)
            except Exception as e:
                logger.error(f"Failed to write fixed code to {full_path}: {e}")
                return False
                
        return False

    def verify_file_dynamic(self, file_path: str, test_command: str = None) -> bool:
        """
        Performs dynamic execution and correction loop in the sandbox.
        If test_command is not provided, tries to run the file directly (python file.py).
        """
        if not self.mcp_toolkit:
            logger.warning("No MCP Toolkit provided. Skipping dynamic verification.")
            return True

        logger.info(f"Starting Dynamic Verification for {file_path}")
        
        # Determine command
        if test_command:
            cmd = test_command
        else:
            # Default to running the file
            cmd = f"python {file_path}"

        for i in range(self.max_retries + 1):
            # Execute in Sandbox
            logger.info(f"Executing: {cmd} (Attempt {i+1})")
            
            # We use the sandbox tool directly
            # The sandbox expects paths relative to the container workspace
            # Assuming the output_dir is mounted to /workspace
            
            exit_code, stdout, stderr = self.mcp_toolkit.sandbox.execute_command(cmd)
            
            if exit_code == 0:
                logger.info(f"Dynamic Verification passed for {file_path}")
                return True
            
            trace = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            logger.warning(f"Execution failed for {file_path}. Exit code: {exit_code}")
            
            if i == self.max_retries:
                logger.error(f"Max retries reached for dynamic verification of {file_path}")
                return False
            
            # Read current code for context
            full_path = os.path.join(self.output_dir, file_path)
            try:
                with open(full_path, 'r') as f:
                    code = f.read()
            except Exception:
                logger.error(f"Could not read file {full_path} for dynamic fix.")
                return False

            # Analyze Trace
            fix_instructions = self.sandbox_agent.analyze_trace(code, trace)
            
            # Apply Fix
            logger.info(f"Applying dynamic fixes to {file_path}...")
            new_code = self.modification_agent.run(code, fix_instructions)
            
            # Save fixed code
            try:
                with open(full_path, 'w') as f:
                    f.write(new_code)
            except Exception as e:
                logger.error(f"Failed to write fixed code to {full_path}: {e}")
                return False
                
        return False

    def verify_codebase(self, blueprint: Blueprint):
        """
        Runs verification on all files in the blueprint.
        """
        logger.info("Starting Codebase Verification Phase")
        
        # 1. Static Pass on all files
        for filename in blueprint.dev_plan:
            spec = blueprint.get_spec(filename)
            if spec:
                self.verify_file_static(filename, spec)
        
        # 2. Dynamic Pass
        # Ideally, we run a test suite. If no test suite, we try to run main files.
        # For this reproduction, we'll try to run files that look like scripts or tests.
        
        # Check for explicit test files in the plan
        test_files = [f for f in blueprint.dev_plan if "test" in f.lower() or "bench" in f.lower()]
        
        if test_files:
            logger.info(f"Found test files: {test_files}. Running dynamic verification on them.")
            for tf in test_files:
                self.verify_file_dynamic(tf)
        else:
            logger.info("No explicit test files found. Attempting to run main entry points.")
            # Try to find main.py or similar
            main_files = [f for f in blueprint.dev_plan if "main" in f.lower()]
            for mf in main_files:
                self.verify_file_dynamic(mf)
