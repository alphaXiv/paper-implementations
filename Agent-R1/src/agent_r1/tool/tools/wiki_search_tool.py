"""
Search tool implementation for simulating internet searches
"""

import time
import random
from typing import Dict, List, Any, Optional
import requests
import os
import json

from agent_r1.tool.base import BaseTool

class WikiSearchTool(BaseTool):
    name = "search"
    description = "Search for information on the internet using Wikipedia as a knowledge source."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
    
    def __init__(self):
        super().__init__()
        self.api_url = os.environ.get("WIKI_SEARCH_API_URL", "http://localhost:8000")
        print(f"[DEBUG] Wiki Search API URL: {self.api_url}")
        
        # 禁用代理，避免代理问题
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        
        # 检查API是否可用
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                print("[DEBUG] Wiki Search API is available")
            else:
                print(f"[WARNING] Wiki Search API health check failed: {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Failed to connect to Wiki Search API: {e}")
    
    def execute(self, args: Dict) -> Dict[str, Any]:
        """
        Execute search query
        
        Args:
            args: Tool parameters, containing:
                - "query": search query string
            
        Returns:
            Dict with format {"content": result_content, "success": bool}
        """
        query = args.get("query", "").strip()
        limit = args.get("limit", 5)
        
        try:
            # 调用API进行搜索
            response = requests.get(
                f"{self.api_url}/search",
                params={"query": query, "top_k": limit}
            )
            
            if response.status_code == 200:
                result = response.json()
                formatted_result = self._format_results(result)
                return {"content": formatted_result, "success": True}
            else:
                error_msg = f"Search API returned error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                print(f"[WARNING] {error_msg}")
                return {"content": error_msg, "success": False}
        except Exception as e:
            error_msg = f"Failed to execute search: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return {"content": error_msg, "success": False}
    
    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        Execute multiple search queries in batch
        
        Args:
            args_list: List of tool parameters
            
        Returns:
            List of dicts with format {"content": result_content, "success": bool}
        """
        # 提取查询和限制
        queries = [args.get("query", "").strip() for args in args_list]
        limits = [5 for args in args_list]
        max_limit = max(limits)  # 使用最大的limit值
        
        try:
            # 调用批量搜索API
            response = requests.post(
                f"{self.api_url}/search",
                json={"queries": queries, "top_k": max_limit}
            )
            
            if response.status_code == 200:
                batch_result = response.json()
                # 为每个查询格式化结果
                results = []
                for i, query_result in enumerate(batch_result["query_results"]):
                    # 限制结果数量为每个查询指定的limit
                    limited_results = {
                        "query": query_result["query"],
                        "results": query_result["results"][:limits[i]]
                    }
                    formatted_result = self._format_results(limited_results)
                    results.append({"content": formatted_result, "success": True})
                return results
            else:
                error_msg = f"Batch search API returned error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                print(f"[WARNING] {error_msg}")
                return [{"content": error_msg, "success": False} for _ in queries]
        except Exception as e:
            error_msg = f"Failed to execute batch search: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return [{"content": error_msg, "success": False} for _ in queries]

    def _format_results(self, api_result) -> str:
        """
        Format API search results for better readability
        
        Args:
            api_result: API response containing search results
            
        Returns:
            Formatted results as a string
        """
        if "error" in api_result:
            return json.dumps(api_result, ensure_ascii=False)
        
        if "query_results" in api_result:
            # 单个查询的结果
            if len(api_result["query_results"]) > 0:
                query_result = api_result["query_results"][0]
                results_list = []
                
                for result in query_result["results"]:
                    # 只提取文档内容，丢弃分数和多余的元数据
                    document = result["document"]
                    clean_result = {
                        "content": document.get("contents", ""),
                        "title": document.get("title", "")
                    }
                    results_list.append(clean_result)
                
                return json.dumps({"results": results_list}, ensure_ascii=False)
            else:
                return json.dumps({"results": []}, ensure_ascii=False)
        elif "results" in api_result:
            # 已经是格式化的结果
            if isinstance(api_result["results"], list):
                clean_results = []
                for result in api_result["results"]:
                    if "document" in result:
                        document = result["document"]
                        clean_result = {
                            "content": document.get("contents", ""),
                            "title": document.get("title", "")
                        }
                        clean_results.append(clean_result)
                    else:
                        # 如果结果已经是处理过的，直接添加
                        clean_results.append(result)
                return json.dumps({"results": clean_results}, ensure_ascii=False)
            else:
                # 结果不是列表，保持原样
                return json.dumps(api_result, ensure_ascii=False)
        else:
            # 单个查询的结果
            results_list = []
            for result in api_result.get("results", []):
                if "document" in result:
                    document = result["document"]
                    clean_result = {
                        "content": document.get("contents", ""),
                        "title": document.get("title", "")
                    }
                    results_list.append(clean_result)
                else:
                    # 如果结果已经是处理过的，直接添加
                    results_list.append(result)
            
            return json.dumps({"results": results_list}, ensure_ascii=False)