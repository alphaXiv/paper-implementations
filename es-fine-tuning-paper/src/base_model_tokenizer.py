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
        required=True,
        help="Directory to save the tokenizer with custom chat template"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the tokenizer with sample messages"
    )
    
    args = parser.parse_args()
    
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
