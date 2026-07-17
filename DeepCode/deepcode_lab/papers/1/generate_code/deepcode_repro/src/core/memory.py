import json
from typing import List, Dict, Set, Optional, Any, Union
from pydantic import BaseModel, Field
from deepcode_repro.src.core.blueprint import FunctionSignature, ClassSignature
from deepcode_repro.src.utils.logger import logger

class DependencyEdges(BaseModel):
    """
    Represents the dependency graph edges for a specific file.
    """
    afferent: List[str] = Field(default_factory=list, description="Incoming dependencies: Files that import this file")
    efferent: List[str] = Field(default_factory=list, description="Outgoing dependencies: Files that this file imports")

class MemoryEntry(BaseModel):
    """
    Represents the summarized state of a generated file (m_t).
    """
    file_path: str
    core_purpose: str
    public_interface: List[Union[FunctionSignature, ClassSignature]] = Field(default_factory=list)
    dependency_edges: DependencyEdges = Field(default_factory=DependencyEdges)
    implementation_summary: str = Field(default="", description="High-level summary of the implementation details")

    def to_text(self) -> str:
        """
        Converts the memory entry to a text representation for LLM context.
        """
        sigs = []
        for sig in self.public_interface:
            if isinstance(sig, FunctionSignature):
                args_str = ", ".join([f"{arg['name']}: {arg['type']}" for arg in sig.args])
                sigs.append(f"Function: {sig.name}({args_str}) -> {sig.return_type}")
            elif isinstance(sig, ClassSignature):
                methods_str = ", ".join([m.name for m in sig.methods])
                sigs.append(f"Class: {sig.name} [Methods: {methods_str}]")
        
        interface_str = "\n  ".join(sigs) if sigs else "None"
        
        return (
            f"File: {self.file_path}\n"
            f"Purpose: {self.core_purpose}\n"
            f"Interface:\n  {interface_str}\n"
            f"Dependencies (Imports): {', '.join(self.dependency_edges.efferent)}\n"
        )

class CodeMem(BaseModel):
    """
    Manages the global state of the generated codebase (CodeMem).
    """
    entries: Dict[str, MemoryEntry] = Field(default_factory=dict)
    
    def add_entry(self, entry: MemoryEntry):
        """
        Update M_t = M_{t-1} U {entry}
        """
        self.entries[entry.file_path] = entry
        logger.info(f"Updated CodeMem with entry for: {entry.file_path}")
        
        # Update afferent edges for files that this file imports
        for dep in entry.dependency_edges.efferent:
            if dep in self.entries:
                if entry.file_path not in self.entries[dep].dependency_edges.afferent:
                    self.entries[dep].dependency_edges.afferent.append(entry.file_path)

    def get_entry(self, file_path: str) -> Optional[MemoryEntry]:
        return self.entries.get(file_path)

    def select_relevant(self, target_file: str, declared_dependencies: List[str]) -> List[MemoryEntry]:
        """
        SelectRelevantMemory(M, target_file): Return subset of M where dependency exists.
        
        Args:
            target_file: The file currently being generated.
            declared_dependencies: List of files the target_file is known to depend on (from Blueprint).
        
        Returns:
            List of MemoryEntry objects relevant to the target_file.
        """
        relevant_memory = []
        
        # 1. Direct dependencies (Explicitly declared in Blueprint)
        for dep_path in declared_dependencies:
            if dep_path in self.entries:
                relevant_memory.append(self.entries[dep_path])
            else:
                logger.warning(f"Dependency {dep_path} for {target_file} not found in CodeMem.")
        
        # 2. Implicit dependencies (Files that might depend on what we are building - context awareness)
        # Usually we care about what we import (efferent), which is covered above.
        # Sometimes we might care about files that will import us (afferent) to match their expectations,
        # but usually those files haven't been generated yet if we follow topological sort.
        
        # 3. Global context (optional: add core utils if they exist and are not explicitly listed)
        # For now, we stick to the explicit dependencies to minimize context window usage.
        
        return relevant_memory

    def to_context_string(self, relevant_entries: List[MemoryEntry]) -> str:
        """
        Formats a list of memory entries into a context string for the LLM.
        """
        if not relevant_entries:
            return "No existing code context available."
            
        context_parts = ["--- EXISTING CODE CONTEXT ---"]
        for entry in relevant_entries:
            context_parts.append(entry.to_text())
        
        return "\n\n".join(context_parts)

    def save(self, path: str):
        """Save CodeMem state to JSON file."""
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))
            
    @classmethod
    def load(cls, path: str) -> 'CodeMem':
        """Load CodeMem state from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except FileNotFoundError:
            return cls()
