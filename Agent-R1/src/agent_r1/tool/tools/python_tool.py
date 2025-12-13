from typing import Dict, List, Any
import os

from agent_r1.tool.base import BaseTool
from sandbox_fusion import set_sandbox_endpoint, run_concurrent, run_code, RunCodeRequest, RunStatus

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

class PythonTool(BaseTool):
    name = "python"
    description = "Python code sandbox, which can be used to execute Python code."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. The Python code should be complete scripts, including necessary imports. IMPORTANT: Use print() statements to output any results you want to see, otherwise they won't be visible.",
            }
        },
        "required": ["code"],
    }

    def __init__(self):
        super().__init__()
        set_sandbox_endpoint('http://localhost:8080')
        self.run_timeout = 10
        self.concurrency = 32
        self.max_attempts = 5

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        batch_code = [args.get("code", "") for args in args_list]
        batch_results = []
        results = run_concurrent(run_code, kwargs=[{"request": RunCodeRequest(run_timeout=self.run_timeout, code=c, language='python'), 'max_attempts': self.max_attempts} for c in batch_code], concurrency=self.concurrency)
        for result in results:
            if result.status == RunStatus.Success:
                if result.run_result and result.run_result.stdout and len(result.run_result.stdout) > 0:
                    batch_results.append({"content": result.run_result.stdout, "success": True})
                else:
                    batch_results.append({"content": "Execution successful but no output", "success": True})
            else:
                error_message = result.message or "Unknown error"
                if result.run_result and result.run_result.stderr:
                    error_message = result.run_result.stderr
                elif result.compile_result and result.compile_result.stderr:
                    error_message = result.compile_result.stderr
                batch_results.append({"content": error_message, "success": False})
        for result in batch_results:
            result['content'] = result['content'].strip()
        return batch_results
    
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        return self.batch_execute([args])[0]