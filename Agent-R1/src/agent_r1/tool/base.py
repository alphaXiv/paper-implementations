from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from agent_r1.tool.utils import is_tool_schema
from PIL import Image
from jsonschema import validate, ValidationError
import torch
class BaseTool(ABC):
    name: str = ''
    description: str = ''
    parameters: dict = {}

    def __init__(self):
        if not self.name:
            raise ValueError('Tool name must be provided')
        if not is_tool_schema({'name': self.name, 'description': self.description, 'parameters': self.parameters}):
            raise ValueError(
                'The parameters, when provided as a dict, must confirm to a valid openai-compatible JSON schema.')

    @abstractmethod
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        pass
    
    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        return [self.execute(args, **kwargs) for args in args_list]
    
    @property
    def tool_info(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }
    
    @property
    def tool_description(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def validate_args(self, args: Dict) -> bool:
        try:
            validate(instance=args, schema=self.parameters)
            return True
        except ValidationError:
            return False

class BaseToolEnv(ABC):
    @abstractmethod
    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        """
        The State Transition Function of the Environment

        Args:
            raw_response: The raw response from the LLM
            
        Returns:
            tool_response: The tool response from the environment
            success: If the tool call is successful
            active: If the trajectory is actives
        """
        pass

    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[bool]], List[bool]]:
        results = [self.step(raw_response) for raw_response in raw_responses]
        return [result[0] for result in results], [result[1] for result in results], [result[2] for result in results]
    
    def process_responses_ids(self, tokenizer, raw_responses_ids: torch.Tensor) -> torch.Tensor:
        return raw_responses_ids

    @abstractmethod
    def stop(self, raw_response: str) -> bool:
        pass

    @abstractmethod
    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        pass
    
    @abstractmethod
    def format_tool_response(self, tool_response: str) -> str:
        pass

    @property
    def system_prompt(self) -> str:
        return ""


class BaseImageToolEnv(BaseToolEnv, ABC):
    @abstractmethod
    def step(self, raw_response: str) -> Tuple[str, List[Image.Image], List[bool], bool]:
        pass
    
    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[Image.Image]], List[List[bool]], List[bool]]:
        results = [self.step(raw_response) for raw_response in raw_responses]
        return [result[0] for result in results], [result[1] for result in results], [result[2] for result in results], [result[3] for result in results]
    
    
