"""
Search tool implementation for simulating internet searches
"""

from typing import Dict, List, Any
import os

from agent_r1.tool.base import BaseTool

import faiss
from FlagEmbedding import FlagAutoModel
import json

class SearchTool(BaseTool):
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
        print("[DEBUG] EMBEDDINGS LOADING")
        
        # Get the absolute path to the data directory
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/corpus/hotpotqa"))
        data_dir = "data/corpus/hotpotqa"
        
        # Load index and corpus using absolute paths
        self.index = faiss.read_index(os.path.join(data_dir, "index.bin"))
        self.model = FlagAutoModel.from_finetuned(
            'BAAI/bge-large-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            devices="cpu",   # if not specified, will use all available gpus or cpu when no gpu available
        )
        self.corpus = []
        with open(os.path.join(data_dir, "hpqa_corpus.jsonl"), "r") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                self.corpus.append(data['title'] + " " + data["text"])
        print("[DEBUG] EMBEDDINGS LOADING END")

    def execute(self, args: Dict) -> Dict[str, Any]:
        """
        Execute search query
        
        Args:
            args: Tool parameters, containing:
                - "query": search query string
            
        Returns:
            Formatted search results
        """
        try:
            query = args["query"]
            embeddings = self.model.encode_queries([query])
            dist, ids = self.index.search(embeddings, 5) # ids: b*5
            result_str = self._format_results(ids[0])
            return {"content": result_str, "success": True}
        except Exception as e:
            return {"content": str(e), "success": False}
    
    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        try:
            queries = [x["query"] for x in args_list]
            embeddings = self.model.encode_queries(queries)
            dist, ids = self.index.search(embeddings, 5) # ids: b*5
            results_str = [self._format_results(ids[i]) for i in range(len(ids))]
            return [{"content": result_str, "success": True} for result_str in results_str]
        except Exception as e:
            return [{"content": str(e), "success": False} for _ in args_list]

    def _format_results(self, results: List) -> str:
        """
        Format search results for better readability
        
        Args:
            results: List of search result List
            
        Returns:
            Formatted results as a string
        """
        results_list = []
        
        for i, result in enumerate(results):
            results_list.append(self.corpus[result])
        
        return json.dumps({"results": results_list})