import argparse
import json
import os
from typing import Any, Dict

from openai import OpenAI

from olmocr.data.renderpdf import render_pdf_to_base64png


def verify_latex_match(
    pdf_path: str,
    page_num: int,
    latex_expression: str,
    model: str = "gpt-4o-2024-08-06",
    temperature: float = 0.1,
    target_longest_image_dim: int = 2048,
) -> Dict[str, Any]:
    """
    Verify if a LaTeX math expression matches what appears in a PDF page.

    Args:
        pdf_path (str): Path to the PDF file
        page_num (int): Page number to check (1-indexed)
        latex_expression (str): LaTeX expression to verify
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
    This is a mathematical expression verification task.
    
    I'm showing you a page from a PDF document containing mathematical expressions.
    
    Please verify if the following LaTeX expression:
    
    {latex_expression}
    
    appears correctly in the document.
    
    Respond with a JSON object containing:
    1. "status": "correct" or "incorrect"
    2. "confidence": a value between 0 and 1 representing your confidence in the answer
    3. "explanation": a brief explanation of why you believe the expression is correct or incorrect
    
    Focus specifically on checking if this exact mathematical expression appears in the document.
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
        "math": latex_expression,
        "status": result.get("status", "unknown"),
        "confidence": result.get("confidence", 0),
        "explanation": result.get("explanation", "No explanation provided"),
    }


def process_jsonl_file(input_jsonl_path: str, output_jsonl_path: str, model: str = "o4-mini-2025-04-16", temperature: float = 0.1) -> None:
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
                    math_expr = entry.get("math")

                    if not all([pdf_path, math_expr]):
                        print(f"Line {line_num}: Skipping entry due to missing required fields")
                        continue

                    print(f"Line {line_num}: Processing: {pdf_path}, page {page_num}")

                    try:
                        result = verify_latex_match(pdf_path=pdf_path, page_num=page_num, latex_expression=math_expr, model=model, temperature=temperature)
                        out_file.write(json.dumps(result) + "\n")
                        processed_count += 1
                    except Exception as e:
                        print(f"Line {line_num}: Error processing {pdf_path}: {str(e)}")
                        error_result = {"pdf": pdf_path, "math": math_expr, "status": "error", "explanation": str(e)}
                        out_file.write(json.dumps(error_result) + "\n")
                        processed_count += 1

                except json.JSONDecodeError:
                    print(f"Line {line_num}: Invalid JSON, skipping")

    print(f"Processed {processed_count} entries. Results saved to {output_jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify LaTeX math expressions in PDFs")
    parser.add_argument("input_jsonl", help="Path to input JSONL file")
    parser.add_argument("output_jsonl", help="Path to output JSONL file")
    parser.add_argument("--model", default="o4-mini-2025-04-16", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for API call")

    args = parser.parse_args()

    process_jsonl_file(input_jsonl_path=args.input_jsonl, output_jsonl_path=args.output_jsonl, model=args.model, temperature=args.temperature)


if __name__ == "__main__":
    main()
