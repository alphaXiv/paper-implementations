from typing import Dict, Any, Optional
import json
import re

from deepcode_repro.src.agents.base import BaseAgent
from deepcode_repro.src.core.blueprint import Blueprint
from deepcode_repro.src.utils.prompts import AgentPrompts
from deepcode_repro.src.utils.logger import logger

class ConceptAgent(BaseAgent):
    """
    Sub-agent responsible for extracting high-level logic and conceptual architecture
    from the research paper.
    """
    def run(self, paper_text: str) -> str:
        logger.info(f"[{self.name}] Extracting conceptual schema...")
        system_prompt = AgentPrompts.CONCEPT_AGENT_SYSTEM
        # We limit paper text if it's too huge, but usually context windows are large enough now.
        # For safety, we might truncate if needed, but let's assume it fits or is handled by the LLM provider.
        user_message = f"Analyze the following research paper and describe the high-level conceptual architecture, core components, and data flow. Focus on the 'what' and 'why'.\n\nPaper Content:\n{paper_text}"
        return self.call_llm(system_prompt, user_message)

class AlgorithmAgent(BaseAgent):
    """
    Sub-agent responsible for extracting mathematical formulas, algorithms, and pseudocode.
    """
    def run(self, paper_text: str) -> str:
        logger.info(f"[{self.name}] Extracting algorithmic schema...")
        system_prompt = AgentPrompts.ALGORITHM_AGENT_SYSTEM
        user_message = f"Analyze the following research paper and extract all mathematical formulations, algorithms, and pseudocode. Provide step-by-step logic and use verbatim LaTeX for equations. Focus on the 'how'.\n\nPaper Content:\n{paper_text}"
        return self.call_llm(system_prompt, user_message)

class PlannerAgent(BaseAgent):
    """
    Sub-agent responsible for merging schemas and synthesizing the Blueprint JSON.
    """
    def run(self, paper_text: str, concept_schema: str, algo_schema: str) -> Blueprint:
        logger.info(f"[{self.name}] Synthesizing Blueprint...")
        system_prompt = AgentPrompts.PLANNING_AGENT_SYSTEM
        
        # Construct the synthesis task
        user_message = f"""
        Synthesize the Implementation Blueprint based on the extracted schemas.
        
        --- ORIGINAL PAPER (Excerpt) ---
        {paper_text[:4000]}... [Truncated for context]
        
        --- CONCEPTUAL SCHEMA ---
        {concept_schema}
        
        --- ALGORITHMIC SCHEMA ---
        {algo_schema}
        
        Generate the full JSON Blueprint containing 'project_name', 'file_hierarchy', 'component_specs', and 'dev_plan'.
        Ensure the JSON is valid and matches the schema definitions provided in the system prompt.
        Return ONLY the JSON object.
        """
        
        response = self.call_llm(system_prompt, user_message)
        
        # Clean and parse JSON
        try:
            # Strip markdown code blocks if present
            clean_json = response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            
            # Attempt to parse
            blueprint_data = json.loads(clean_json)
            
            # Validate and convert to Blueprint object
            # We use from_json which expects a string, or we can construct directly if we have the dict
            # Blueprint.from_json expects a JSON string.
            return Blueprint.from_json(json.dumps(blueprint_data))
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Blueprint JSON: {e}")
            logger.debug(f"Raw response: {response}")
            raise ValueError("LLM failed to generate valid JSON for Blueprint") from e
        except Exception as e:
            logger.error(f"Error creating Blueprint object: {e}")
            raise e

class PlanningSwarm:
    """
    Orchestrates the Concept, Algorithm, and Planning agents to produce a repository blueprint.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.concept_agent = ConceptAgent("ConceptAgent", config)
        self.algorithm_agent = AlgorithmAgent("AlgorithmAgent", config)
        self.planner_agent = PlannerAgent("PlanningAgent", config)

    def plan(self, paper_text: str) -> Blueprint:
        """
        Executes the planning pipeline:
        1. Extract Conceptual Schema
        2. Extract Algorithmic Schema
        3. Synthesize Blueprint
        """
        logger.info("Starting Planning Swarm...")
        
        # Phase 1: Parallel Extraction (Sequential here for simplicity)
        # In a real async system, these could run in parallel
        concept_schema = self.concept_agent.run(paper_text)
        algo_schema = self.algorithm_agent.run(paper_text)
        
        # Phase 2: Synthesis
        blueprint = self.planner_agent.run(paper_text, concept_schema, algo_schema)
        
        logger.info(f"Blueprint synthesis complete for project: {blueprint.project_name}")
        return blueprint
