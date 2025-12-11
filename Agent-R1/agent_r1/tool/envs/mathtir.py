from torch import Tensor
from agent_r1.tool.base import BaseToolEnv, BaseTool
from verl.utils.torch_functional import pad_2d_list_to_length
from typing import List, Tuple, Any
import re

class MathTIREnv(BaseToolEnv):
    def __init__(self, tools: List[BaseTool], max_tool_response_length: int):
        self.tool = tools[0]
        assert self.tool.name == "python"
        self.max_tool_response_length = max_tool_response_length

    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        code = self.extract_tool_calls(raw_response)
        if len(code) == 0:
            return "", [], False
        code = code[0]
        tool_response, tool_success = self.tool.execute({"code": code})
        tool_response = self.format_tool_response([tool_response])
        return tool_response, [tool_success], True

    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[bool]], List[bool]]:
        batch_tool_response = [""] * len(raw_responses)
        batch_tool_successes = [[]] * len(raw_responses)
        batch_active = [True] * len(raw_responses)
        codes = []
        for i, raw_response in enumerate(raw_responses):
            code = self.extract_tool_calls(raw_response)
            if len(code) == 0:
                batch_tool_response[i] = ""
                batch_tool_successes[i] = []
                batch_active[i] = False
                continue
            codes.append({"code": code[0]})
        results = self.tool.batch_execute(codes)
        i = 0
        for j in len(range(raw_responses)):
            if batch_active[j]:
                result = results[i]
                batch_tool_response[j] = self.format_tool_response([result["content"]])
                batch_tool_successes[j] = [result["success"]]
                i += 1
        return batch_tool_response, batch_tool_successes, batch_active
    
    def process_responses_ids(self, tokenizer, raw_responses_ids: Tensor) -> Tensor:
        def process_response_ids(raw_response_ids):
            complete_response = self.tokenizer.decode(raw_response_ids)
            if not re.search(r"```python(.*)```", complete_response):
                return raw_response_ids
            for i in range(len(raw_response_ids) - 1):
                if raw_response_ids[i] == tokenizer.eos_token_id:
                    return raw_response_ids
                current_response = self.tokenizer.decode(raw_response_ids[:i+1])
                if re.search(r"```python(.*)```", current_response):
                    return raw_response_ids[:i+1]
            return raw_response_ids
        responses_ids = [process_response_ids(raw_response_ids) for raw_response_ids in raw_responses_ids.tolist()]
        responses_ids = pad_2d_list_to_length(responses_ids, tokenizer.pad_token_id, max_length=raw_responses_ids.size(1)).to(raw_responses_ids.device)
        return responses_ids

    def stop(self, raw_response: str) -> bool:
        tool_calls = self.extract_tool_calls(raw_response)
        if len(tool_calls) == 0:
            return True
        else:
            return False
        
    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        """
        extract the code after "```python", and before "```"
        """
        code = ''
        start = False
        for line in raw_response.split('\n'):
            if line.startswith('```python') or line.endswith('```python'):
                code += '\n# ========\n'
                start = True
            elif line.startswith('```') and not line.startswith('```python'):
                start = False
            elif start:
                code += line + '\n'
        if start or len(code) == 0:
            # the code is incomplete
            return []
        return [code]
        
    def format_tool_response(self, tool_responses: List[str]) -> str:
        if len(tool_responses) == 0:
            return ""
        if len(tool_responses[0]) > self.max_tool_response_length:
            tool_responses[0] = tool_responses[0][:self.max_tool_response_length] + "..."
        return "\n```output\n" + tool_responses[0] + "\n```\n"