import os
from typing import Dict, Any, List, Optional
from deepcode_repro.src.agents.base import BaseAgent
from deepcode_repro.src.core.blueprint import Blueprint, ComponentSpecification
from deepcode_repro.src.core.memory import CodeMem, MemoryEntry, DependencyEdges
from deepcode_repro.src.core.rag_engine import CodeRAG
from deepcode_repro.src.utils.prompts import AgentPrompts
from deepcode_repro.src.utils.logger import logger
from deepcode_repro.src.utils.mcp_tools import MCPToolkit

class CodingAgent(BaseAgent):
    """
    Agent responsible for generating implementation code for a specific file
    based on the blueprint specification, memory context, and RAG knowledge.
    """
    def __init__(self, config: Dict[str, Any], mcp_toolkit: Optional[MCPToolkit] = None):
        super().__init__(name="CodingAgent", config=config, mcp_toolkit=mcp_toolkit)

    def run(self, 
            file_name: str, 
            spec: ComponentSpecification, 
            memory_context: str, 
            rag_context: str = "") -> str:
        """
        Generates code for the target file.
        """
        logger.info(f"Generating code for {file_name}...")
        
        # Construct the prompt
        spec_str = spec.model_dump_json(indent=2)
        user_message = AgentPrompts.get_coding_task(
            file_name=file_name,
            spec=spec_str,
            context=memory_context,
            rag_data=rag_context
        )
        
        # Call LLM
        response = self.call_llm(
            system_prompt=AgentPrompts.CODING_AGENT_SYSTEM,
            user_message=user_message
        )
        
        # Extract code from markdown blocks if present
        code = self._extract_code(response)
        return code

    def _extract_code(self, text: str) -> str:
        """Extracts code from markdown code blocks."""
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text.strip()


class SummarizationAgent(BaseAgent):
    """
    Agent responsible for analyzing generated code and creating a MemoryEntry
    summary for the CodeMem graph.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="SummarizationAgent", config=config)

    def run(self, file_path: str, code: str) -> MemoryEntry:
        """
        Summarizes the code into a MemoryEntry.
        """
        logger.info(f"Summarizing {file_path} for CodeMem...")
        
        user_message = AgentPrompts.get_summarization_task(code=code)
        
        response = self.call_llm(
            system_prompt=AgentPrompts.SUMMARIZATION_AGENT_SYSTEM,
            user_message=user_message
        )
        
        # Parse the response into a MemoryEntry
        # The prompt asks for a JSON structure. We need to parse it.
        try:
            # Clean up potential markdown wrapping
            json_str = response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            
            # We expect the LLM to return a structure compatible with MemoryEntry
            # However, MemoryEntry has specific fields. 
            # Let's assume the prompt guides the LLM to output the correct JSON structure.
            # We might need to map it manually if the LLM output is slightly off, 
            # but for now we'll try direct parsing or construction.
            
            import json
            data = json.loads(json_str)
            
            # Construct MemoryEntry
            # Ensure dependency_edges is properly formatted
            deps = data.get("dependency_edges", {})
            edges = DependencyEdges(
                afferent=deps.get("afferent", []),
                efferent=deps.get("efferent", [])
            )
            
            entry = MemoryEntry(
                file_path=file_path,
                core_purpose=data.get("core_purpose", "No purpose provided"),
                public_interface=data.get("public_interface", []), # This might need more parsing if it's complex objects
                dependency_edges=edges
            )
            return entry
            
        except Exception as e:
            logger.error(f"Failed to parse summary for {file_path}: {e}")
            # Return a fallback entry
            return MemoryEntry(
                file_path=file_path,
                core_purpose="Summary generation failed",
                public_interface=[],
                dependency_edges=DependencyEdges(afferent=[], efferent=[])
            )


class CodingSwarm:
    """
    Orchestrates the stateful generation loop.
    """
    def __init__(self, 
                 config: Dict[str, Any], 
                 memory: CodeMem, 
                 rag_engine: Optional[CodeRAG] = None,
                 output_dir: str = "data/output_repos"):
        self.config = config
        self.memory = memory
        self.rag_engine = rag_engine
        self.output_dir = output_dir
        
        self.coding_agent = CodingAgent(config)
        self.summarization_agent = SummarizationAgent(config)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_codebase(self, blueprint: Blueprint) -> None:
        """
        Executes the generation loop for the entire blueprint.
        """
        logger.info(f"Starting codebase generation for project: {blueprint.project_name}")
        
        # Iterate through the development plan (topological sort order)
        for file_path in blueprint.dev_plan:
            self._generate_file(file_path, blueprint)
            
        logger.info("Codebase generation complete.")

    def _generate_file(self, file_path: str, blueprint: Blueprint) -> None:
        """
        Generates a single file, saves it, and updates memory.
        """
        logger.info(f"Processing target file: {file_path}")
        
        # 1. Get Component Specification
        spec = blueprint.get_spec(file_path)
        if not spec:
            logger.warning(f"No specification found for {file_path}. Skipping.")
            return

        # 2. Select Relevant Memory
        # We use the dependencies declared in the blueprint spec to filter memory
        relevant_entries = self.memory.select_relevant(file_path, spec.dependencies)
        memory_context = self.memory.to_context_string(relevant_entries)
        
        # 3. Retrieve RAG Context (if enabled)
        rag_context = ""
        if self.rag_engine:
            # Simple heuristic: always check, or use should_retrieve
            # The prompt context + target file is used to decide
            if self.rag_engine.should_retrieve(spec.core_purpose, file_path):
                logger.info(f"Retrieving RAG context for {file_path}")
                # Query based on the spec description and purpose
                query_text = f"{file_path}: {spec.core_purpose}. {spec.algorithmic_details}"
                results = self.rag_engine.query(query_text, n_results=3)
                # Format results
                rag_context = "\n".join([f"Snippet from {r['metadata'].get('source_id', 'unknown')}:\n{r['content']}" for r in results])

        # 4. Generate Code
        code = self.coding_agent.run(
            file_name=file_path,
            spec=spec,
            memory_context=memory_context,
            rag_context=rag_context
        )
        
        # 5. Save Code to Disk
        full_path = os.path.join(self.output_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)
        logger.info(f"Saved {file_path}")
        
        # 6. Summarize and Update Memory
        memory_entry = self.summarization_agent.run(file_path, code)
        self.memory.add_entry(memory_entry)
