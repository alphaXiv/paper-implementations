import json
import numpy as np
# encode
from FlagEmbedding import FlagAutoModel
import faiss
import os

if __name__ == "__main__":

 
    corpus = []
    with open("../../../data/corpus/hotpotqa/hpqa_corpus.jsonl") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data["title"] + " " + data["text"])


    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-large-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        devices="cuda:0",   # if not specified, will use all available gpus or cpu when no gpu available
    )

    embeddings = model.encode_corpus(corpus)
    #save
    np.save("../../../data/corpus/hotpotqa/hpqa_corpus.npy", embeddings)

    corpus_numpy = np.load("../../../data/corpus/hotpotqa/hpqa_corpus.npy")
    dim = corpus_numpy.shape[-1]

    corpus_numpy = corpus_numpy.astype(np.float32)
    
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, '../../../data/corpus/hotpotqa/index.bin')