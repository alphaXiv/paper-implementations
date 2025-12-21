from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field
import json

class FunctionSignature(BaseModel):
    """Represents a function signature in the public interface."""
    name: str
    args: List[str]
    return_type: str
    description: Optional[str] = None

class ClassSignature(BaseModel):
    """Represents a class structure in the public interface."""
    name: str
    methods: List[FunctionSignature] = []
    description: Optional[str] = None

class ComponentSpecification(BaseModel):
    """
    Detailed specification for a single file/component.
    Merged result of Concept Agent and Algorithm Agent.
    """
    filename: str
    core_purpose: str = Field(..., description="High-level logic and responsibility of this component")
    algorithmic_details: Optional[str] = Field(None, description="Verbatim LaTeX equations, pseudocode, or step-by-step logic")
    dependencies: List[str] = Field(default_factory=list, description="List of other files or external libraries this component depends on")
    public_interface: List[Union[FunctionSignature, ClassSignature]] = Field(default_factory=list, description="Expected classes and functions")
    implementation_notes: Optional[str] = Field(None, description="Specific instructions for the coding agent")

class FileNode(BaseModel):
    """Represents a node in the file hierarchy tree."""
    name: str
    type: str = Field(..., pattern="^(file|directory)$")
    children: Optional[List['FileNode']] = None
    description: Optional[str] = None

class Blueprint(BaseModel):
    """
    The Master Blueprint (B) synthesized by the Planning Agent.
    Contains the architecture, detailed specs, and execution plan.
    """
    project_name: str
    file_hierarchy: List[FileNode] = Field(..., description="Tree structure of the repository")
    component_specs: Dict[str, ComponentSpecification] = Field(..., description="Map of filename to detailed specification")
    dev_plan: List[str] = Field(..., description="Ordered list of filenames to generate (topological sort)")

    def to_json(self) -> str:
        """Serialize blueprint to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Blueprint':
        """Load blueprint from JSON string."""
        return cls.model_validate_json(json_str)

    def get_spec(self, filename: str) -> Optional[ComponentSpecification]:
        """Retrieve specification for a specific file."""
        return self.component_specs.get(filename)

    def save(self, path: str):
        """Save blueprint to a file."""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'Blueprint':
        """Load blueprint from a file."""
        with open(path, 'r') as f:
            content = f.read()
        return cls.from_json(content)
