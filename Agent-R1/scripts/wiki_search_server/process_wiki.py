import json
import numpy as np
# encode
from FlagEmbedding import FlagAutoModel
import faiss
import os
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import argparse

def create_index(embeddings, index_type="IVF4096,Flat", nprobe=64):
    """
    Create a FAISS index with the specified configuration.
    
    Args:
        embeddings: The embeddings to add to the index
        index_type: The type of index to create
        nprobe: Number of clusters to probe at search time (for IVF indexes)
        
    Returns:
        The created FAISS index
    """
    dim = embeddings.shape[1]
    embeddings = embeddings.astype(np.float32)
    
    # Normalize vectors to unit length for cosine similarity
    faiss.normalize_L2(embeddings)
    
    if index_type.startswith("IVF"):
        # For IVF indexes, we need to train the index
        print(f"Creating {index_type} index...")
        index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index
        print("Training index...")
        index.train(embeddings)
        
        # Set the number of clusters to probe at search time
        if hasattr(index, 'nprobe'):
            index.nprobe = nprobe
            print(f"Setting nprobe to {nprobe}")
    else:
        # For other index types
        print(f"Creating {index_type} index...")
        index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)
    
    # Add vectors to the index
    print("Adding vectors to index...")
    index.add(embeddings)
    
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Wikipedia data and create FAISS index")
    parser.add_argument("--index_type", type=str, default="IVF4096,PQ96", 
                        help="FAISS index type (e.g., Flat, IVF4096,Flat, IVF4096,PQ96, HNSW32)")
    parser.add_argument("--nprobe", type=int, default=64, 
                        help="Number of clusters to probe at search time (for IVF indexes)")
    parser.add_argument("--skip_processing", action="store_true", 
                        help="Skip dataset processing and embedding generation")
    args = parser.parse_args()

    os.makedirs("../../data/corpus", exist_ok=True)
    os.makedirs("../../data/corpus/wiki", exist_ok=True)
    
    if not args.skip_processing:
        # load wiki dataset
        print("Loading Wikipedia dataset...")
        dataset = load_dataset("NeuML/wikipedia-20250123", split="train")
        
        chunks = []
        data_list = []
        print("Splitting text into chunks...")
        for item in tqdm(dataset):
            id = item["id"]
            url = item["url"]
            title = item["title"]
            text = item["text"]
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = splitter.split_text(text)
            for chunk in text_chunks:
                data_list.append({
                    "id": id,
                    "url": url,
                    "title": title,
                    "text": chunk
                })
            chunks.extend(text_chunks)

        print("Saving corpus to JSONL...")
        with open("../../data/corpus/wiki/wiki_corpus.jsonl", "w") as f:
            for data in data_list:
                f.write(json.dumps(data) + "\n")
        
        print(f"Total chunks: {len(chunks)}")

        print("Generating embeddings...")
        model = FlagAutoModel.from_finetuned(
            'BAAI/bge-large-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            # devices="cuda:0",   # if not specified, will use all available gpus or cpu when no gpu available
        )

        embeddings = model.encode_corpus(chunks)
        
        print("Saving embeddings...")
        np.save("../../data/corpus/wiki/wiki_corpus.npy", embeddings)
    
    print("Loading embeddings...")
    corpus_numpy = np.load("../../data/corpus/wiki/wiki_corpus.npy")
    print(f"Loaded embeddings with shape: {corpus_numpy.shape}")
    
    # Create and save the index
    index = create_index(corpus_numpy, index_type=args.index_type, nprobe=args.nprobe)
    
    print(f"Saving index with type {args.index_type}...")
    index_filename = f"../../data/corpus/wiki/wiki_index_{args.index_type.replace(',', '_')}.bin"
    faiss.write_index(index, index_filename)
    print(f"Index saved to {index_filename}")
    
    # Save a small metadata file with the index configuration
    with open(f"{index_filename}.meta", "w") as f:
        json.dump({
            "index_type": args.index_type,
            "nprobe": args.nprobe,
            "embedding_shape": list(corpus_numpy.shape),
            "metric": "inner_product"
        }, f)
    
    print("Done!")