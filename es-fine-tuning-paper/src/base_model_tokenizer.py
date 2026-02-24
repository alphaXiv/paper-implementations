"""
Custom tokenizer wrapper for base models (Llama, Qwen) with simple chat template.
This adds a chat template to base model tokenizers for training without instruction-tuned models.

Template format: "Question: {input} Answer: Let's think step by step."
"""

from transformers import AutoTokenizer
from typing import List, Dict, Union, Optional
import os


class BaseModelTokenizer:
    """Wrapper class that adds a simple chat template to base model tokenizers."""
    
    # Simple chat template for base models
    CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}Question: {{ message['content'] }}
Answer: Let's think step by step.{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"""
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize tokenizer with custom chat template.
        
        Args:
            model_path: Path to the base model (e.g., 'meta-llama/Llama-3.2-3B' or 'Qwen/Qwen2.5-3B')
            **kwargs: Additional arguments to pass to AutoTokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        
        # Add chat template if not present
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = self.CHAT_TEMPLATE
            print(f"Added custom chat template to tokenizer from {model_path}")
        else:
            # Override existing chat template with our simple one
            print(f"Overriding existing chat template for base model training")
            self.tokenizer.chat_template = self.CHAT_TEMPLATE
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print("Added [PAD] as pad_token")
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the tokenizer with the custom chat template."""
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        print(f"Saved tokenizer with custom chat template to {save_directory}")
    
    def __getattr__(self, name):
        """Delegate all other attribute access to the underlying tokenizer."""
        return getattr(self.tokenizer, name)
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper callable like the original tokenizer."""
        return self.tokenizer(*args, **kwargs)


def create_base_tokenizer(model_path: str, save_path: Optional[str] = None, **kwargs) -> BaseModelTokenizer:
    """
    Create and optionally save a base model tokenizer with custom chat template.
    
    Args:
        model_path: Path to the base model
        save_path: Optional path to save the tokenizer
        **kwargs: Additional arguments for AutoTokenizer
    
    Returns:
        BaseModelTokenizer instance
    """
    tokenizer = BaseModelTokenizer(model_path, **kwargs)
    
    if save_path:
        tokenizer.save_pretrained(save_path)
    
    return tokenizer


def test_model_generation(model_path: str, data_path: str, num_samples: int = 30):
    """
    Test model generation with countdown dataset samples, similar to VERL's approach.
    
    Args:
        model_path: Path to the model
        data_path: Path to parquet data file (e.g., train.parquet)
        num_samples: Number of samples to generate
    """
    from transformers import AutoModelForCausalLM
    import torch
    import pandas as pd
    
    print(f"\n{'='*60}")
    print(f"Testing Model Generation (VERL-style)")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Samples: {num_samples}")
    
    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_parquet(data_path)
    print(f"Total samples in dataset: {len(df)}")
    
    # Take first num_samples
    samples = df.head(num_samples)
    
    # Load tokenizer with custom chat template
    print("\n[2/4] Loading tokenizer...")
    tokenizer = BaseModelTokenizer(model_path)
    
    # Load model
    print("[3/4] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    
    print(f"\n[4/4] Generating answers for {num_samples} samples...")
    print(f"{'='*60}\n")
    
    # Generate answers
    results = []
    for idx, row in samples.iterrows():
        # Extract the user question from the prompt
        prompt_messages = row['prompt']
        # Find the user message (skip system message if present)
        user_content = None
        for msg in prompt_messages:
            if msg['role'] == 'user':
                user_content = msg['content']
                break
        
        if user_content is None:
            print(f"Warning: No user message found in sample {idx}, skipping...")
            continue
        
        # Format message with chat template
        messages = [{"role": "user", "content": user_content}]
        
        # Apply chat template and tokenize (VERL-style)
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(device)
        
        # Generate using model.generate (VERL-style)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part (after the prompt)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        answer = generated_text[len(prompt_text):].strip()
        
        results.append({
            "question": user_content,
            "prompt": prompt_text,
            "answer": answer,
            "full_output": generated_text
        })
        
        # Print progress
        print(f"Sample {idx + 1}/{num_samples}")
        print(f"Q: {user_content}")
        print(f"A: {answer}")
        print(f"{'-'*60}\n")
    
    print(f"\n{'='*60}")
    print(f"✓ Generation test completed successfully!")
    print(f"Total samples: {len(results)}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create custom tokenizer for base models with simple chat template"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to base model (e.g., meta-llama/Llama-3.2-3B, Qwen/Qwen2.5-3B)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Directory to save the tokenizer with custom chat template"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the tokenizer with sample messages"
    )
    parser.add_argument(
        "--test_generation",
        action="store_true",
        help="Test model generation with multiple samples (VERL-style)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/countdown-0.1/train.parquet",
        help="Path to parquet data file (default: data/countdown-0.1/train.parquet)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of samples to generate (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Test generation mode
    if args.test_generation:
        results = test_model_generation(args.model_path, args.data_path, args.num_samples)
    else:
        # Create tokenizer
        print(f"\nCreating tokenizer from: {args.model_path}")
        tokenizer = create_base_tokenizer(args.model_path, args.save_path)
        
        if args.test:
            # Test the tokenizer
            print("\n" + "="*50)
            print("Testing tokenizer with sample messages")
            print("="*50)
            
            messages = [
                {
                    "role": "user",
                    "content": "Using the numbers [3, 5, 7, 10], create an equation that equals 24."
                }
            ]
            
            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            print("\nFormatted text:")
            print(formatted_text)
            
            # Tokenize
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            
            print(f"\nTokens shape: {tokens.shape}")
            print(f"Number of tokens: {tokens.shape[1]}")
            
            # Decode back
            decoded = tokenizer.decode(tokens[0])
            print(f"\nDecoded text:")
            print(decoded)
            
            print("\n✓ Tokenizer test completed successfully!")
