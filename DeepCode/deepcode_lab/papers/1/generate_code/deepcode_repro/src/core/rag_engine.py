import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any, Tuple
from deepcode_repro.src.utils.logger import logger
from deepcode_repro.src.core.document_parser import Segment

class CodeRAG:
    """
    CodeRAG Engine: Handles indexing and retrieval of semantic context.
    
    Implements Component 4: CodeRAG Engine.
    Purpose: Conditional knowledge injection for underspecified designs.
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "deepcode_context"):
        """
        Initialize the CodeRAG engine with a ChromaDB client.
        
        Args:
            persist_directory: Path to store the vector database.
            collection_name: Name of the collection to use.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"CodeRAG initialized with collection '{collection_name}' at '{persist_directory}'")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None

    def index_segments(self, segments: List[Segment], source_id: str = "doc"):
        """
        Index document segments into the vector store.
        
        Args:
            segments: List of Segment objects from DocumentSegmenter.
            source_id: Identifier for the source document (e.g., filename).
        """
        if not self.collection:
            logger.warning("CodeRAG collection not initialized. Skipping indexing.")
            return

        ids = []
        documents = []
        metadatas = []
        
        for i, seg in enumerate(segments):
            # Create a unique ID for each segment
            seg_id = f"{source_id}_seg_{i}"
            
            # Prepare metadata
            meta = {
                "source": source_id,
                "header": seg.header,
                "level": seg.level,
                "index": i
            }
            # Add any existing metadata from the segment
            if seg.metadata:
                for k, v in seg.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
            
            ids.append(seg_id)
            documents.append(seg.to_text())
            metadatas.append(meta)
            
        if ids:
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Indexed {len(ids)} segments from {source_id}")
            except Exception as e:
                logger.error(f"Failed to index segments: {e}")

    def query(self, query_text: str, n_results: int = 3, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant segments for a given query.
        
        Args:
            query_text: The search query (e.g., code snippet or description).
            n_results: Number of results to return.
            filter_metadata: Optional dictionary to filter results by metadata.
            
        Returns:
            List of dictionaries containing 'content', 'metadata', and 'distance'.
        """
        if not self.collection:
            logger.warning("CodeRAG collection not initialized. Returning empty results.")
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Parse results into a cleaner format
            parsed_results = []
            if results['ids']:
                # Chroma returns lists of lists (one list per query)
                ids = results['ids'][0]
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                distances = results['distances'][0] if results['distances'] else [0.0] * len(ids)
                
                for id_, doc, meta, dist in zip(ids, docs, metas, distances):
                    parsed_results.append({
                        "id": id_,
                        "content": doc,
                        "metadata": meta,
                        "distance": dist
                    })
            
            logger.info(f"Retrieved {len(parsed_results)} results for query: '{query_text[:50]}...'")
            return parsed_results
            
        except Exception as e:
            logger.error(f"Failed to query CodeRAG: {e}")
            return []

    def clear_index(self):
        """Reset the collection."""
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.get_or_create_collection(self.collection_name)
                logger.info("CodeRAG index cleared.")
            except Exception as e:
                logger.error(f"Failed to clear index: {e}")

    def should_retrieve(self, context_text: str, target_file: str) -> bool:
        """
        Determine if retrieval is necessary (Adaptive Retrieval Step 1).
        
        Algorithm: r_t = delta(Context, target_file)
        
        Note: In a full implementation, this might use an LLM to decide.
        Here we implement a heuristic: if the target file description or context 
        contains keywords indicating external dependencies or complex logic 
        that might be in the knowledge base, return True.
        
        For now, we default to True if the query is substantial, 
        or we can make it always True for this reproduction to ensure RAG is used.
        """
        # Simple heuristic: always retrieve if we have a target file
        # In a real agentic loop, the agent might decide this.
        return bool(target_file)
