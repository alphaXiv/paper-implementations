from agent_r1.tool.base import BaseToolEnv, BaseTool
from typing import List, Tuple, Any

class ReToolEnv(BaseToolEnv):
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
        for j in range(len(raw_responses)):
            if batch_active[j]:
                result = results[i]
                batch_tool_response[j] = self.format_tool_response([result["content"]])
                batch_tool_successes[j] = [result["success"]]
                i += 1
        return batch_tool_response, batch_tool_successes, batch_active

    def stop(self, raw_response: str) -> bool:
        tool_calls = self.extract_tool_calls(raw_response)
        if len(tool_calls) == 0:
            return True
        else:
            return False
        
    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        """
        extract the code after "<code>", and before "</code>"
        """
        code = ''
        start = False
        for line in raw_response.split('\n'):
            if line.startswith('<code>'):
                code += '\n# ========\n'
                start = True
            elif line.startswith('</code>'):
                start = False
            elif start:
                if line.startswith('```'):
                    continue
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
        return "\n<interpreter>\n" + tool_responses[0] + "\n</interpreter>\n"