#!/usr/bin/env python3
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import openai
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64png


def process_test_case(case: Dict[str, Any], client, pdf_dir: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Send a request to GPT-4 asking if the before and after text appear in the same region.
    Include the PDF image in the prompt.

    Args:
        case: A test case from the JSONL file
        client: The OpenAI client
        pdf_dir: Directory containing PDF files
        model: The model to use

    Returns:
        The original case with the added response field
    """
    before_text = case["before"]
    after_text = case["after"]
    pdf_path = os.path.join(pdf_dir, case["pdf"])
    page_num = case["page"]

    try:
        # Render the PDF page to a base64-encoded PNG image
        image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num)

        # Create messages with both text and image
        messages = [
            {"role": "system", "content": "You are an AI assistant analyzing text from PDFs."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Does the text in the 'before' field and the 'after' field appear in the same region of the page? "
                            f"Look at the PDF image and determine if these texts are located near each other or in completely "
                            f"different parts of the page. Different regions could be the captions for different images, or inside of different insets or tables. However, appearing the same column of text, or in the naturally flowing next column of text is close enough.\n\n"
                            f"Before: {before_text}\n\n"
                            f"After: {after_text}\n\n"
                            f"Respond with 'YES' if they appear in the same region or column, and 'NO' if they appear in "
                            f"different regions. Then explain your reasoning in 1-2 sentences."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            },
        ]

        # Call the API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=300,
        )

        # Add GPT-4's response to the case
        case_with_response = case.copy()
        case_with_response["gpt4_response"] = response.choices[0].message.content
        return case_with_response
    except Exception as e:
        # In case of error, return the original case with an error message
        case_with_response = case.copy()
        case_with_response["gpt4_response"] = f"ERROR: {str(e)}"
        # Print the error for debugging
        print(f"Error processing {case.get('id', 'unknown')}: {str(e)}")
        return case_with_response


def process_jsonl_file(input_file: str, output_file: str, api_key: str, pdf_dir: str, num_workers: int = 8, model: str = "gpt-4o") -> None:
    """
    Process each line in the JSONL file by sending requests to GPT-4 in parallel.

    Args:
        input_file: Path to the input JSONL file
        output_file: Path to write the output JSONL file with responses
        api_key: OpenAI API key
        pdf_dir: Directory containing PDF files
        num_workers: Number of parallel workers
        model: The model to use
    """
    # Read all test cases from the input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Parse each line to get test cases
    test_cases = []
    for line in lines:
        if line.strip():
            test_cases.append(json.loads(line))

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Process test cases in parallel
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_case = {executor.submit(process_test_case, case, client, pdf_dir, model): case for case in test_cases}

        # Process results as they complete
        for future in tqdm(as_completed(future_to_case), total=len(test_cases), desc="Processing test cases"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                case = future_to_case[future]
                print(f"Error processing case {case.get('id', 'unknown')}: {str(e)}")
                # Add failed case with error message
                case["gpt4_response"] = f"PROCESSING_ERROR: {str(e)}"
                results.append(case)

    # Filter for cases where GPT-4 responded with "NO"
    no_responses = [result for result in results if "gpt4_response" in result and result["gpt4_response"].startswith("NO")]

    # Write filtered results to output file
    with open(output_file, "w") as f:
        for result in no_responses:
            f.write(json.dumps(result) + "\n")

    print(f"Processed {len(results)} test cases. Found {len(no_responses)} cases with 'NO' responses. Results written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process multi_column.jsonl with GPT-4 to check text regions")
    parser.add_argument("--input", default="/home/ubuntu/olmocr/olmOCR-bench/bench_data/multi_column.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output", default="/home/ubuntu/olmocr/olmOCR-bench/bench_data/multi_column_gpt4_regions.jsonl", help="Path to output JSONL file")
    parser.add_argument("--pdf-dir", default="/home/ubuntu/olmocr/olmOCR-bench/bench_data/pdfs", help="Directory containing the PDF files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use")
    parser.add_argument("--api-key", help="OpenAI API key (if not provided, uses OPENAI_API_KEY env var)")

    args = parser.parse_args()

    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided either via --api-key or OPENAI_API_KEY environment variable")

    # Verify that the PDF directory exists
    if not os.path.isdir(args.pdf_dir):
        raise ValueError(f"PDF directory {args.pdf_dir} does not exist")

    process_jsonl_file(input_file=args.input, output_file=args.output, api_key=api_key, pdf_dir=args.pdf_dir, num_workers=args.workers, model=args.model)


if __name__ == "__main__":
    main()
