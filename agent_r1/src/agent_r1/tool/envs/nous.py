from agent_r1.tool.base import BaseToolEnv, BaseTool
from typing import List, Dict, Tuple, Any
import re
import json

class NousToolEnv(BaseToolEnv):
    def __init__(self, tools: List[BaseTool], max_tool_response_length: int):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.tool_call_start = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_response_start = "<tool_response>"
        self.tool_response_end = "</tool_response>"
        self.eos_token = "<|im_end|>"
        self.parallel_tool_calls = False
        self.max_tool_response_length = max_tool_response_length
    
    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        tool_calls = self.extract_tool_calls(raw_response)
        if len(tool_calls) == 0:
            return "", [], False
        if not self.parallel_tool_calls:
            tool_calls = [tool_calls[0]]
        tool_responses = []
        tool_successes = []
        for tool_call in tool_calls:
            if tool_call is None:
                tool_responses.append("Error: JSONDecodeError")
                tool_successes.append(False)
            else:
                if "name" not in tool_call:
                    tool_responses.append("Error: No tool name")
                    tool_successes.append(False)
                else:
                    tool_name = tool_call["name"]
                    if tool_name not in self.tool_map:
                        tool_responses.append("Error: ToolNotFoundError")
                        tool_successes.append(False)
                    else:
                        tool = self.tool_map[tool_name]
                        if not tool.validate_args(tool_call["arguments"]):
                            tool_responses.append("Error: Invalid tool arguments")
                            tool_successes.append(False)
                        else:
                            tool_result = tool.execute(tool_call["arguments"])
                            tool_responses.append(tool_result["content"])
                            tool_successes.append(tool_result["success"])
        tool_response = self.format_tool_response(tool_responses)
        return tool_response, tool_successes, True

    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[bool]], List[bool]]:
        batch_tool_responses = [[]] * len(raw_responses)
        batch_tool_successes = [[]] * len(raw_responses)
        batch_active = [True] * len(raw_responses)
        success_tool_calls_arguments = {} # batch 内成功的工具调用。key: tool_name，value: [arguments]
        success_tool_calls_index = {} # batch 内成功的工具调用。key: tool_name，value: [(i,j)]
        for i, raw_response in enumerate(raw_responses):
            tool_calls = self.extract_tool_calls(raw_response)
            if len(tool_calls) == 0:
                batch_tool_successes[i] = []
                batch_active[i] = False
                batch_tool_responses[i] = []
                continue

            if not self.parallel_tool_calls:
                tool_calls = [tool_calls[0]]
            tool_responses = []
            tool_successes = []
            for j, tool_call in enumerate(tool_calls):
                if tool_call is None:
                    tool_responses.append("Error: JSONDecodeError")
                    tool_successes.append(False)
                else:
                    if "name" not in tool_call:
                        tool_responses.append("Error: No tool name")
                        tool_successes.append(False)
                    elif "arguments" not in tool_call:
                        tool_responses.append("Error: No tool arguments")
                        tool_successes.append(False)
                    else:
                        tool_name = tool_call["name"]
                        if tool_name not in self.tool_map:
                            tool_responses.append("Error: ToolNotFoundError")
                            tool_successes.append(False)
                        else:
                            tool = self.tool_map[tool_name]
                            if not tool.validate_args(tool_call["arguments"]):
                                tool_responses.append("Error: Invalid tool arguments")
                                tool_successes.append(False)
                            else:
                                # 默认success_tool_calls[tool_name]
                                if tool_name not in success_tool_calls_arguments:
                                    success_tool_calls_arguments[tool_name] = []
                                    success_tool_calls_index[tool_name] = []
                                tool_responses.append("Executing...")
                                tool_successes.append(False)
                                success_tool_calls_arguments[tool_name].append(tool_call["arguments"])
                                success_tool_calls_index[tool_name].append((i,j))
            batch_tool_responses[i] = tool_responses
            batch_tool_successes[i] = tool_successes
        
        # batch excute
        for tool_name, args_list in success_tool_calls_arguments.items():
            tool = self.tool_map[tool_name]
            batch_results = tool.batch_execute(args_list)
            for batch_result, (i,j) in zip(batch_results, success_tool_calls_index[tool_name]):
                assert batch_tool_responses[i][j] == "Executing..."
                batch_tool_responses[i][j] = batch_result["content"]
                batch_tool_successes[i][j] = batch_result["success"]
        
        batch_tool_responses_ = []
        for i, tool_responses in enumerate(batch_tool_responses):
            if batch_active[i]:
                assert len(batch_tool_responses[i]) > 0
                batch_tool_responses_.append(self.format_tool_response(tool_responses))
            else:
                batch_tool_responses_.append("")
        
        return batch_tool_responses_, batch_tool_successes, batch_active

    def stop(self, raw_response: str) -> bool:
        tool_calls = self.extract_tool_calls(raw_response)
        if len(tool_calls) == 0:
            return True
        else:
            return False
        
    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        tool_calls = []
        pattern = re.compile(f"{re.escape(self.tool_call_start)}(.*?){re.escape(self.tool_call_end)}", re.DOTALL)
        for tool_call in re.findall(pattern, raw_response):
            try:
                tool_call = json.loads(tool_call)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                tool_calls.append(None)
        
        return tool_calls
        
    def format_tool_response(self, tool_responses: List[str]) -> str:
        tool_message = "<|im_end|>\n<|im_start|>user\n"
        for i, tool_response in enumerate(tool_responses):
            if len(tool_response) > self.max_tool_response_length:
                tool_response = tool_response[:self.max_tool_response_length] + "..."
            tool_message += f"<tool_response>\n{tool_response}\n</tool_response>"
            if i < len(tool_responses) - 1:
                tool_message += "\n"
        tool_message += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return tool_message
        