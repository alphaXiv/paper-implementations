import os
import datasets
from pathlib import Path

if __name__ == '__main__':
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent
    DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'data' / 'processed')
    DATASET_NAME = 'openai/gsm8k'

    # Load dataset
    raw_train_dataset = datasets.load_dataset(DATASET_NAME, "main", split="train")
    raw_test_dataset = datasets.load_dataset(DATASET_NAME, "main", split="test")

    instruction_following = "Reason step by step, and put your final answer on a new line in the following format:\n\n#### <number>"

    # Process function to add data_source and append instruction_following to question
    def process_example(example):
        example['data_source'] = DATASET_NAME
        example['question'] = example['question'] + "\n\n" + instruction_following
        return example

    # Apply processing to train dataset first
    processed_train_dataset = raw_train_dataset.map(process_example)
    processed_test_dataset = raw_test_dataset.map(process_example)

    # Split train_dataset: 128 examples for validation, rest for training
    # Use train_test_split with specific seed for reproducibility
    split_datasets = processed_train_dataset.train_test_split(test_size=128, seed=42)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test']
    test_dataset = processed_test_dataset
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    print(train_dataset[0])
    print(val_dataset[0])
    print(test_dataset[0])

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    train_dataset.to_parquet(os.path.join(DEFAULT_OUTPUT_DIR, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(DEFAULT_OUTPUT_DIR, 'val.parquet'))
    test_dataset.to_parquet(os.path.join(DEFAULT_OUTPUT_DIR, 'test.parquet'))