#!/usr/bin/env python3
"""
mine_old_scans_math.py - Extract mathematical equations from PDF documents.

This script:
1. Takes a folder containing PDF documents as input
2. For each PDF, extracts a random page and renders it to an image
3. Uses Gemini to identify mathematical equations in the rendered image
4. Creates a test file asserting that the equation text should be present
5. Extracts the page from the PDF and saves it to an output folder

Usage:
  python mine_old_scans_math.py --input_dir path/to/pdf_folder --output_dir path/to/output --api_key your_gemini_api_key
"""

import argparse
import base64
import json
import os
import random
from typing import List, Optional

import pypdf
from google import genai
from google.genai import types
from tqdm import tqdm

from olmocr.bench.tests import TextPresenceTest, save_tests
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.filter import PdfFilter


def extract_page_from_pdf(input_path: str, output_path: str, page_num: int) -> bool:
    """
    Extract a specific page from a PDF and save it as a new PDF.

    Args:
        input_path: Path to the input PDF
        output_path: Path to save the extracted page
        page_num: The page number to extract (0-indexed)

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read the input PDF
        reader = pypdf.PdfReader(input_path)

        # Check if page number is valid
        if page_num >= len(reader.pages):
            print(f"Page number {page_num} out of range for {input_path} with {len(reader.pages)} pages")
            return False

        # Create a new PDF with just the selected page
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[page_num])

        # Write the output PDF
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        return True
    except Exception as e:
        print(f"Error extracting page {page_num} from {input_path}: {str(e)}")
        raise


def detect_equations(pdf_path: str, page_num: int, api_key: str) -> Optional[List[str]]:
    """
    Use Gemini to detect mathematical equations in a rendered PDF page.

    Args:
        pdf_path: Path to the PDF file
        page_num: The page number to analyze (0-indexed)
        api_key: Gemini API key

    Returns:
        Optional[List[str]]: List of detected equations, or None if detection failed
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = "gemini-2.0-flash"

    # Render the PDF page as an image
    try:
        image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num + 1, target_longest_image_dim=2048)  # render_pdf_to_base64png is 1-indexed
    except Exception as e:
        print(f"Error rendering PDF page: {str(e)}")
        return None

    image_part = types.Part(inline_data=types.Blob(mime_type="image/png", data=base64.b64decode(image_base64)))

    contents = [
        types.Content(
            role="user",
            parts=[
                image_part,
                types.Part.from_text(
                    text="""Please extract the mathematical equations from the document without omission. Always output the mathematical equations as Latex escaped with $$. Do not hallucinate"""
                ),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "equations": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                ),
            },
        ),
    )

    response = client.models.generate_content(model=model, contents=contents, config=generate_content_config)

    assert len(response.candidates) > 0, "No candidates found"
    assert response.candidates[0].finish_reason == types.FinishReason.STOP, "Finish reason was not STOP, likely a processing error or repetition failure"

    data = json.loads(response.candidates[0].content.parts[0].text)

    return data.get("equations", [])


def process_pdf(pdf_path: str, output_dir: str, api_key: str, tests: List[TextPresenceTest], max_pages_per_pdf: int = 10) -> None:
    """
    Process a single PDF, extracting equations from multiple pages.

    Args:
        pdf_path: Path to the PDF
        output_dir: Directory for output files
        api_key: Gemini API key
        tests: List to append tests to
        max_pages_per_pdf: Maximum number of pages to process per PDF
    """
    # Extract filename from path
    pdf_filename = os.path.basename(pdf_path)

    pdf_filter = PdfFilter()

    if pdf_filter.filter_out_pdf(pdf_path):
        print("Filtering out", pdf_filename)
        return

    try:
        # Read the PDF to get the number of pages
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"PDF {pdf_filename} has no pages")
            return

        # Get all pages and shuffle them to select a random subset
        all_pages = list(range(num_pages))
        random.shuffle(all_pages)

        # Take only the specified maximum number of pages
        pages_to_process = all_pages[: min(max_pages_per_pdf, num_pages)]

        processed_pages = 0

        for page_num in pages_to_process:
            # Detect equations
            equations = detect_equations(pdf_path, page_num, api_key)

            # Only keep equations that are non-empty
            equations = [eq for eq in equations if len(eq.strip()) > 3]

            if not equations:
                print(f"No equations detected in {pdf_filename} page {page_num+1}")
                continue

            # Extract the page and save to output dir
            pdf_basename = os.path.splitext(pdf_filename)[0]
            output_pdf_path = os.path.join(output_dir, "pdfs", f"{pdf_basename}_pg{page_num+1}.pdf")

            extract_page_from_pdf(pdf_path, output_pdf_path, page_num)

            # Create tests for each equation
            for i, equation in enumerate(equations):
                test_id = f"{pdf_basename}_pg{page_num+1}_equation_{i:02d}"
                test = TextPresenceTest(
                    id=test_id,
                    pdf=f"{pdf_basename}_pg{page_num+1}.pdf",
                    page=1,  # The extracted PDF has only one page
                    type="present",
                    text=equation,
                    max_diffs=0,
                )
                tests.append(test)

            print(f"Processed {pdf_filename} page {page_num+1}, found {len(equations)} equations")
            processed_pages += 1

        print(f"Completed processing {processed_pages} pages from {pdf_filename}")

    except Exception as e:
        print(f"Error processing {pdf_filename}: {str(e)}")


def get_pdf_files_from_directory(directory: str) -> List[str]:
    """
    Get a list of all PDF files in a directory.

    Args:
        directory: Path to the directory containing PDFs

    Returns:
        List[str]: List of full paths to PDF files
    """
    pdf_files = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):
                pdf_files.append(full_path)

    return pdf_files


def main():
    parser = argparse.ArgumentParser(description="Extract mathematical equations from PDF documents")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted pages and tests")
    parser.add_argument("--api_key", help="Gemini API key (if not provided, will use GEMINI_API_KEY environment variable)")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Gemini API key not provided. Use --api_key or set GEMINI_API_KEY environment variable.")
        return

    # Create directories
    os.makedirs(os.path.join(args.output_dir, "pdfs"), exist_ok=True)

    # Get PDF files from input directory
    pdf_files = get_pdf_files_from_directory(args.input_dir)

    if not pdf_files:
        print(f"No PDF files found in {args.input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files in input directory")

    # Process each PDF
    tests = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        process_pdf(pdf_path, args.output_dir, api_key, tests)

        # Save tests after each PDF to avoid losing data in case of crashes
        if tests:
            save_tests(tests, os.path.join(args.output_dir, "equation_tests.jsonl"))

        # if len(tests) > 100:
        #     break

    print(f"Saved {len(tests)} tests to {os.path.join(args.output_dir, 'equation_tests.jsonl')}")


if __name__ == "__main__":
    main()
