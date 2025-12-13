from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@lru_cache()
def load_coherency_model(model_name: str = "HuggingFaceTB/SmolLM-135M"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()  # Set the model to evaluation mode

    return tokenizer, model


def get_document_coherency(text: str) -> float:
    """
    Calculates the coherency of a document based on the log likelihood of its tokens.
    Handles texts longer than the model's maximum token limit by splitting them into chunks.

    Args:
        text (str): The input text to evaluate.

    Returns:
        float: The average log likelihood per token as a measure of coherency.
    """
    tokenizer, model = load_coherency_model()

    # Determine the model's maximum number of tokens
    max_length = tokenizer.model_max_length - 1
    # Some tokenizers have a default value indicating no limit; use model config if so
    if max_length > 1_000_000:
        max_length = model.config.max_position_embeddings

    # Tokenize the entire text
    tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)

    total_log_likelihood = 0.0
    total_tokens = 0

    # Split tokens into chunks that fit within the model's max length
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i : i + max_length]
        inputs = chunk.unsqueeze(0)  # Add batch dimension

        # Move inputs to CPU (ensure compatibility)
        inputs = {k: v.cpu() for k, v in {"input_ids": inputs}.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            # Compute log likelihood for the chunk
            log_likelihood = -outputs.loss.item() * chunk.size(0)
            total_log_likelihood += log_likelihood
            total_tokens += chunk.size(0)

    # Calculate the average log likelihood per token
    avg_log_likelihood = total_log_likelihood / total_tokens if total_tokens > 0 else 0.0

    return avg_log_likelihood
