import argparse
import json
import os
from typing import Any, Dict

from openai import OpenAI

from olmocr.data.renderpdf import render_pdf_to_base64png


def verify_header_footer_match(
    pdf_path: str,
    page_num: int,
    hea_foo_text: str,
    model: str,
    temperature: float = 0.1,
    target_longest_image_dim: int = 2048,
) -> Dict[str, Any]:
    """
    Verify if a headers and footers matches what appears in a PDF page.

    Args:
        pdf_path (str): Path to the PDF file
        page_num (int): Page number to check (1-indexed)
        model (str): OpenAI model to use
        temperature (float): Temperature for API call
        target_longest_image_dim (int): Target dimension for the image

    Returns:
        Dict with verification result
    """
    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("You must specify an OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
    This is a header and footer verification task.
    
    I'm showing you a page from a PDF document containing headers and footers text.
    
    Please verify if the headers or footers are exactly matches the below text.
    
    {hea_foo_text}
    
    Respond with a JSON object containing:
    1. "status": "correct" or "incorrect"
    2. "confidence": a value between 0 and 1 representing your confidence in the answer
    3. "explanation": a brief explanation of why you believe the text is correct or incorrect
    
    Focus specifically on checking if this exact header or footer expression appears in the document.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        # temperature=temperature,
        response_format={"type": "json_object"},
        # max_tokens=1000,
    )
    raw_response = response.choices[0].message.content
    result = json.loads(raw_response)

    return {
        "pdf": pdf_path,
        "math": hea_foo_text,
        "status": result.get("status", "unknown"),
        "confidence": result.get("confidence", 0),
        "explanation": result.get("explanation", "No explanation provided"),
    }


def process_jsonl_file(input_jsonl_path: str, output_jsonl_path: str, model: str = "o3-2025-04-16", temperature: float = 0.1) -> None:
    """
    Process a JSONL file containing math expressions to verify.

    Args:
        input_jsonl_path (str): Path to input JSONL file
        output_jsonl_path (str): Path to output JSONL file
        model (str): OpenAI model to use
        temperature (float): Temperature for API call
    """
    processed_count = 0

    with open(output_jsonl_path, "w") as out_file:
        with open(input_jsonl_path, "r") as in_file:
            for line_num, line in enumerate(in_file, 1):
                try:
                    entry = json.loads(line.strip())

                    pdf_path = entry.get("pdf")
                    page_num = entry.get("page", 1)
                    text_expr = entry.get("text")

                    if not all([pdf_path, text_expr]):
                        print(f"Line {line_num}: Skipping entry due to missing required fields")
                        continue

                    print(f"Line {line_num}: Processing: {pdf_path}, page {page_num}")

                    try:
                        result = verify_header_footer_match(pdf_path=pdf_path, page_num=page_num, hea_foo_text=text_expr, model=model, temperature=temperature)
                        out_file.write(json.dumps(result) + "\n")
                        processed_count += 1
                    except Exception as e:
                        print(f"Line {line_num}: Error processing {pdf_path}: {str(e)}")
                        error_result = {"pdf": pdf_path, "text": text_expr, "status": "error", "explanation": str(e)}
                        out_file.write(json.dumps(error_result) + "\n")
                        processed_count += 1

                except json.JSONDecodeError:
                    print(f"Line {line_num}: Invalid JSON, skipping")

    print(f"Processed {processed_count} entries. Results saved to {output_jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify headers footers expressions in PDFs")
    parser.add_argument("input_jsonl", help="Path to input JSONL file")
    parser.add_argument("output_jsonl", help="Path to output JSONL file")
    parser.add_argument("--model", default="o3-2025-04-16", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for API call")

    args = parser.parse_args()

    process_jsonl_file(input_jsonl_path=args.input_jsonl, output_jsonl_path=args.output_jsonl, model=args.model, temperature=args.temperature)


if __name__ == "__main__":
    main()
