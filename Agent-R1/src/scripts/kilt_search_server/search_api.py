import json
import numpy as np
import faiss
import time
import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from FlagEmbedding import FlagAutoModel
from datasets import load_dataset
from tqdm import tqdm

app = FastAPI(title="Wiki Search API", description="API for searching Wikipedia using FAISS index")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
index = None
corpus = None

class SearchQuery(BaseModel):
    queries: List[str]
    top_k: int = 10

class SearchResult(BaseModel):
    score: float
    document: Dict[str, Any]

class QueryResult(BaseModel):
    query: str
    results: List[SearchResult]

class SearchResponse(BaseModel):
    query_results: List[QueryResult]
    total_time: float
    search_time: float

def load_index(index_path):
    """Load a FAISS index from disk"""
    print(f"Loading index from {index_path}...")
    index = faiss.read_index(index_path)
    
    # Load metadata if available
    meta_path = f"{index_path}.meta"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "nprobe" in meta and hasattr(index, "nprobe"):
                index.nprobe = meta["nprobe"]
                print(f"Setting nprobe to {meta['nprobe']}")
    
    return index

def load_corpus(dataset_name="corag/kilt-corpus", split_name="train"):
    """Load the corpus from dataset"""
    print(f"Loading corpus from {dataset_name}...")
    # corpus = []
    dataset = load_dataset(dataset_name, split=split_name)
    # for item in tqdm(dataset):
    #     corpus.append(item)
    return dataset

def search(queries: List[str], top_k=10):
    """
    Search the corpus for the most similar documents to the queries
    
    Args:
        queries: List of query strings
        top_k: Number of results to return for each query
        
    Returns:
        SearchResponse object with results for each query
    """
    global model, index, corpus
    
    if model is None or index is None or corpus is None:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    # Encode the queries
    start_time = time.time()
    query_embeddings = model.encode_queries(queries)
    
    # 确保query_embeddings是numpy数组并且是float32类型
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    
    # Normalize the query embeddings for cosine similarity
    faiss.normalize_L2(query_embeddings)
    
    # Search the index
    search_start = time.time()
    scores, indices = index.search(query_embeddings, top_k)
    search_end = time.time()
    
    # Get the results for each query
    query_results = []
    for q_idx, query in enumerate(queries):
        results = []
        for i, idx in enumerate(indices[q_idx]):
            if idx != -1:  # -1 means no result found
                results.append(SearchResult(
                    score=float(scores[q_idx][i]),
                    document=corpus[int(idx)]
                ))
        
        query_results.append(QueryResult(
            query=query,
            results=results
        ))
    
    end_time = time.time()
    
    return SearchResponse(
        query_results=query_results,
        total_time=end_time - start_time,
        search_time=search_end - search_start
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup"""
    global model, index, corpus
    
    # 配置路径
    index_path = os.environ.get("INDEX_PATH", "../../data/corpus/kilt/kilt_index_IVF16384_PQ64.bin")
    
    # 加载模型
    print("Loading model...")
    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-large-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        device="cpu"
    )
    
    # 加载索引
    index = load_index(index_path)
    
    # 加载语料库
    corpus = load_corpus()
    
    print("Search engine initialized successfully")

@app.post("/search", response_model=SearchResponse)
async def api_search(search_query: SearchQuery):
    """Search the corpus for the most similar documents to the queries"""
    return search(search_query.queries, search_query.top_k)

@app.get("/search", response_model=SearchResponse)
async def api_search_get(
    query: str = Query(..., description="The query to search for (for multiple queries, use POST method)"),
    top_k: int = Query(10, description="Number of results to return")
):
    """Search the corpus for the most similar documents to the query (GET method)"""
    return search([query], top_k)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or index is None or corpus is None:
        raise HTTPException(status_code=503, detail="Search engine not fully initialized")
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Wiki Search API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000) 