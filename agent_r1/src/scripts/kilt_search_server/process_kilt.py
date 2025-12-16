import json
import numpy as np
# encode
from FlagEmbedding import FlagAutoModel
import faiss
import os
from datasets import load_dataset
from tqdm import tqdm
import argparse
import torch

def create_index(embeddings, index_type="IVF4096,Flat", nprobe=64, use_gpu=True, gpu_ids=None, 
                 hnsw_ef_construction=200, hnsw_ef_search=128):
    """
    Create a FAISS index with the specified configuration.
    
    Args:
        embeddings: The embeddings to add to the index
        index_type: The type of index to create
        nprobe: Number of clusters to probe at search time (for IVF indexes)
        use_gpu: Whether to use GPU for index building
        gpu_ids: List of GPU IDs to use. If None, use all available GPUs
        hnsw_ef_construction: HNSW efConstruction parameter (size of the dynamic list for HNSW construction)
        hnsw_ef_search: HNSW efSearch parameter (size of the dynamic list for HNSW search)
        
    Returns:
        The created FAISS index
    """
    print("Starting index creation...")
    dim = embeddings.shape[1]

    # Normalize vectors to unit length for cosine similarity
    print("Normalize vectors to unit length for cosine similarity")
    faiss.normalize_L2(embeddings)
    
    # Initialize GPU resources first before normalization
    gpu_resources = []
    num_gpus = faiss.get_num_gpus()
    print(f"Num GPUs: {num_gpus}")
    
    if use_gpu and num_gpus > 0:
        # Use specific GPUs if provided, otherwise use all available
        if gpu_ids is None:
            gpu_ids = list(range(num_gpus))
        else:
            # Ensure all provided GPU IDs are valid
            gpu_ids = [id for id in gpu_ids if id < num_gpus]
            
        if not gpu_ids:
            print("No valid GPU IDs provided, falling back to CPU")
        else:
            print(f"Using {len(gpu_ids)} GPUs (IDs: {gpu_ids}) for index building")
            for gpu_id in gpu_ids:
                res = faiss.StandardGpuResources()
                gpu_resources.append(res)
    
    if index_type.startswith("IVF"):
        # For IVF indexes, we need to train the index
        print(f"Creating {index_type} index...")
        index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)
        
        # Convert to GPU index for training if available
        gpu_index = index
        
        if gpu_resources:
            if len(gpu_resources) > 1:
                # Multi-GPU setup
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True  # Shard the index across GPUs
                gpu_index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index, co)
                print(f"Distributed index across {len(gpu_resources)} GPUs for training")
            else:
                # Single GPU setup
                gpu_index = faiss.index_cpu_to_gpu(gpu_resources[0], 0, index)
                print("Transferred index to GPU for training")
        
        # Train the index
        print("Training index...")
        batch_size = 1000000  # Adjust based on GPU memory
        if embeddings.shape[0] > batch_size and hasattr(gpu_index, 'train_batch'):
            print(f"Training in batches of {batch_size}")
            for i in range(0, embeddings.shape[0], batch_size):
                end = min(i + batch_size, embeddings.shape[0])
                print(f"Training batch {i//batch_size + 1}/{(embeddings.shape[0]-1)//batch_size + 1}")
                gpu_index.train_batch(embeddings[i:end])
        else:
            gpu_index.train(embeddings)
        
        # Convert back to CPU if we used GPU
        if gpu_resources:
            if len(gpu_resources) > 1 or not isinstance(gpu_index, faiss.IndexCPU):
                index = faiss.index_gpu_to_cpu(gpu_index)
                print("Transferred index back to CPU after training")
            else:
                index = gpu_index
        else:
            index = gpu_index
        
        # Set the number of clusters to probe at search time
        if hasattr(index, 'nprobe'):
            index.nprobe = nprobe
            print(f"Setting nprobe to {nprobe}")
    elif index_type.startswith("HNSW"):
        # For HNSW indexes, we set specific HNSW parameters
        print(f"Creating {index_type} index with HNSW parameters: efConstruction={hnsw_ef_construction}")
        index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)
        
        # Set HNSW parameters if applicable
        if hasattr(index, 'hnsw'):
            index.hnsw.efConstruction = hnsw_ef_construction
            index.hnsw.efSearch = hnsw_ef_search
            print(f"Set HNSW parameters: efConstruction={hnsw_ef_construction}, efSearch={hnsw_ef_search}")
        
        # HNSW is typically built on CPU as it doesn't benefit as much from GPU
        print("HNSW indices are typically built on CPU for better quality")
    else:
        # For other index types
        print(f"Creating {index_type} index...")
        index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)
        
        # Convert to GPU index for faster processing if available and not HNSW
        # (HNSW typically doesn't benefit as much from GPU for construction)
        if gpu_resources and not index_type.startswith("HNSW"):
            if len(gpu_resources) > 1:
                # Multi-GPU setup for adding vectors
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index, co)
                print(f"Using {len(gpu_resources)} GPUs for vector addition")
            else:
                # Single GPU
                index = faiss.index_cpu_to_gpu(gpu_resources[0], 0, index)
                print("Using GPU for vector addition")
    
    # Add vectors to the index
    print("Adding vectors to index...")
    batch_size = 1000000  # Adjust based on GPU memory
    if embeddings.shape[0] > batch_size:
        # HNSW requires adding all vectors at once
        if index_type.startswith("HNSW"):
            print(f"HNSW index requires adding all vectors at once ({embeddings.shape[0]} vectors)")
            index.add(embeddings)
        else:
            print(f"Adding vectors in batches of {batch_size}")
            for i in range(0, embeddings.shape[0], batch_size):
                end = min(i + batch_size, embeddings.shape[0])
                print(f"Adding batch {i//batch_size + 1}/{(embeddings.shape[0]-1)//batch_size + 1}")
                index.add(embeddings[i:end])
    else:
        index.add(embeddings)
    
    # Convert back to CPU for storage if we used GPU
    if gpu_resources and (len(gpu_resources) > 1 or faiss.is_gpu_index(index)):
        index = faiss.index_gpu_to_cpu(index)
        print("Transferred final index back to CPU for storage")
    
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Wikipedia data and create FAISS index")
    parser.add_argument("--index_type", type=str, default="HNSW64", 
                        help="FAISS index type (e.g., Flat, IVF4096,Flat, IVF4096,PQ96, HNSW32)")
    parser.add_argument("--nprobe", type=int, default=64, 
                        help="Number of clusters to probe at search time (for IVF indexes)")
    parser.add_argument("--skip_processing", action="store_true", default=True, 
                        help="Skip dataset processing and embedding generation")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="Use GPU for index building if available")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, use all available GPUs")
    parser.add_argument("--batch_size", type=int, default=1000000,
                        help="Batch size for training and adding vectors to the index")
    # HNSW-specific parameters
    parser.add_argument("--hnsw_ef_construction", type=int, default=200,
                        help="HNSW efConstruction parameter - size of the dynamic list for HNSW construction (default: 200)")
    parser.add_argument("--hnsw_ef_search", type=int, default=128,
                        help="HNSW efSearch parameter - size of the dynamic list for HNSW search (default: 128)")
    args = parser.parse_args()

    os.makedirs("../../data/corpus", exist_ok=True)
    os.makedirs("../../data/corpus/kilt", exist_ok=True)
    
    # Parse GPU IDs if provided
    gpu_id_list = None
    if args.gpu_ids:
        gpu_id_list = [int(id.strip()) for id in args.gpu_ids.split(',')]
    
    if not args.skip_processing:
        # load wiki dataset
        print("Loading Kilt dataset...")
        dataset = load_dataset("corag/kilt-corpus", split="train")
        chunks = []
        for item in tqdm(dataset):
            chunks.append(item['contents'])

        print("Generating embeddings...")
        # For embedding generation, we use the first GPU if use_gpu is True
        # or all GPUs for maximum throughput if supported by the model
        device_ids = None
        if args.use_gpu and torch.cuda.is_available():
            if gpu_id_list:
                # If specific GPUs are specified, use only the first one for embedding
                # This could be extended to support multi-GPU embedding if the model supports it
                device_ids = f"cuda:{gpu_id_list[0]}"
            else:
                device_ids = "cuda:0"  # Use first GPU by default
        
        model = FlagAutoModel.from_finetuned(
            'BAAI/bge-large-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            devices="cuda:0" if args.use_gpu and torch.cuda.is_available() else None,
        )

        embeddings = model.encode_corpus(chunks)
        
        print("Saving embeddings...")
        np.save("../../data/corpus/kilt/kilt_corpus.npy", embeddings)
    
    print("Loading embeddings...")
    embeddings = np.load("../../data/corpus/kilt/kilt_corpus.npy")
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    print(f"Embeddings dtype before conversion: {embeddings.dtype}")
    embeddings = embeddings.astype(np.float32)
    
    # Create and save the index
    index = create_index(embeddings, index_type=args.index_type, nprobe=args.nprobe, 
                        use_gpu=args.use_gpu, gpu_ids=gpu_id_list,
                        hnsw_ef_construction=args.hnsw_ef_construction,
                        hnsw_ef_search=args.hnsw_ef_search)
    
    print(f"Saving index with type {args.index_type}...")
    index_filename = f"../../data/corpus/kilt/kilt_index_{args.index_type.replace(',', '_')}.bin"
    faiss.write_index(index, index_filename)
    print(f"Index saved to {index_filename}")
    
    # Save a small metadata file with the index configuration
    with open(f"{index_filename}.meta", "w") as f:
        json.dump({
            "index_type": args.index_type,
            "nprobe": args.nprobe,
            "embedding_shape": list(embeddings.shape),
            "metric": "inner_product",
            "built_with_gpu": args.use_gpu,
            # Add HNSW parameters to metadata if using HNSW
            "hnsw_ef_construction": args.hnsw_ef_construction if args.index_type.startswith("HNSW") else None,
            "hnsw_ef_search": args.hnsw_ef_search if args.index_type.startswith("HNSW") else None
        }, f)
    
    print("Done!")