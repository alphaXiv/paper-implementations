#!/usr/bin/env python3
"""
mine_long_tiny_text.py - Extract long text from PDF documents.

This script:
1. Takes a folder containing PDF documents as input
2. For each PDF, extracts random pages and renders them to images
3. Uses Gemini to identify text in the rendered images
4. Creates test files asserting that the text should be present
5. Extracts the pages from the PDF and saves them to an output folder

Usage:
  python mine_long_tiny_text.py --input_dir path/to/pdf_folder --output_dir path/to/output --api_key your_gemini_api_key
"""

import argparse
import base64
import json
import os
import random
from typing import List

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
        return False  # Return False instead of raising to continue processing other PDFs


def detect_long_text(pdf_path: str, page_num: int, api_key: str) -> List[str]:
    """
    Use Gemini to detect long text in a rendered PDF page.

    Args:
        pdf_path: Path to the PDF file
        page_num: The page number to analyze (0-indexed)
        api_key: Gemini API key

    Returns:
        List[str]: List of detected text, empty list if detection failed
    """
    try:
        client = genai.Client(
            api_key=api_key,  # Use the provided API key instead of environment variable
        )
        model = "gemini-2.0-flash"

        # Render the PDF page as an image
        try:
            image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num + 1, target_longest_image_dim=2048)  # render_pdf_to_base64png is 1-indexed
        except Exception as e:
            print(f"Error rendering PDF page {page_num+1} from {pdf_path}: {str(e)}")
            return []

        image_part = types.Part(inline_data=types.Blob(mime_type="image/png", data=base64.b64decode(image_base64)))

        contents = [
            types.Content(
                role="user",
                parts=[
                    image_part,
                    types.Part.from_text(
                        text="""Extract and display all text from the document without any omissions. The documents may be in a multi-column format, so handle that carefully. If the words are less than 9, combine text from the next line. I don't want all the text, just randomly pick few sentences. Do not summarize, abbreviate, or hallucinate any content."""
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

        if len(response.candidates) == 0:
            print(f"No candidates found for {pdf_path} page {page_num+1}")
            return []

        if response.candidates[0].finish_reason != types.FinishReason.STOP:
            print(f"Finish reason was not STOP for {pdf_path} page {page_num+1}, likely a processing error")
            return []

        try:
            data = json.loads(response.candidates[0].content.parts[0].text)
            return data.get("equations", [])
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response for {pdf_path} page {page_num+1}")
            return []

    except Exception as e:
        print(f"Error detecting text in {pdf_path} page {page_num+1}: {str(e)}")
        return []


def process_pdf(
    pdf_path: str, output_dir: str, api_key: str, tests: List[TextPresenceTest], max_pages_per_pdf: int = 20, force_processing: bool = True
) -> bool:
    """
    Process a single PDF, extracting text from multiple pages.

    Args:
        pdf_path: Path to the PDF
        output_dir: Directory for output files
        api_key: Gemini API key
        tests: List to append tests to
        max_pages_per_pdf: Maximum number of pages to process per PDF
        force_processing: If True, process PDF even if it would normally be filtered out

    Returns:
        bool: True if PDF was processed successfully
    """
    # Extract filename from path
    pdf_filename = os.path.basename(pdf_path)

    pdf_filter = PdfFilter()

    if not force_processing and pdf_filter.filter_out_pdf(pdf_path):
        print(f"Filtering out {pdf_filename} (use --force_processing to override)")
        return False

    try:
        # Read the PDF to get the number of pages
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"PDF {pdf_filename} has no pages")
            return False

        # Get all pages and shuffle them to select a random subset
        all_pages = list(range(num_pages))
        random.shuffle(all_pages)

        # Take only the specified maximum number of pages
        pages_to_process = all_pages[: min(max_pages_per_pdf, num_pages)]

        processed_pages = 0
        pdf_processed = False  # Flag to track if at least one page was processed

        for page_num in pages_to_process:
            # Detect text
            text_sections = detect_long_text(pdf_path, page_num, api_key)

            # Only keep text sections that are non-empty
            text_sections = [text for text in text_sections if len(text.strip()) > 3]

            # Extract the page regardless of text detection
            pdf_basename = os.path.splitext(pdf_filename)[0]
            output_pdf_path = os.path.join(output_dir, "pdfs", f"{pdf_basename}_pg{page_num+1}.pdf")

            # Extract the page regardless of text detection
            page_extracted = extract_page_from_pdf(pdf_path, output_pdf_path, page_num)
            if page_extracted:
                pdf_processed = True

            if not text_sections:
                print(f"No text detected in {pdf_filename} page {page_num+1} - creating empty test")
                # Create a placeholder test to ensure the PDF is represented
                test_id = f"{pdf_basename}_pg{page_num+1}_no_text"
                test = TextPresenceTest(
                    id=test_id,
                    pdf=f"{pdf_basename}_pg{page_num+1}.pdf",
                    page=1,  # The extracted PDF has only one page
                    type="present",
                    text="No text detected",  # Placeholder text
                    max_diffs=0,
                )
                tests.append(test)
            else:
                # Create tests for each text section
                for i, text_section in enumerate(text_sections):
                    test_id = f"{pdf_basename}_pg{page_num+1}_text_{i:02d}"
                    test = TextPresenceTest(
                        id=test_id,
                        pdf=f"{pdf_basename}_pg{page_num+1}.pdf",
                        page=1,  # The extracted PDF has only one page
                        type="present",
                        text=text_section,
                        max_diffs=0,
                    )
                    tests.append(test)

            print(f"Processed {pdf_filename} page {page_num+1}, found {len(text_sections)} text sections")
            processed_pages += 1

        print(f"Completed processing {processed_pages} pages from {pdf_filename}")
        return pdf_processed

    except Exception as e:
        print(f"Error processing {pdf_filename}: {str(e)}")
        return False


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
    parser = argparse.ArgumentParser(description="Extract long text from PDF documents")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted pages and tests")
    parser.add_argument("--api_key", help="Gemini API key (if not provided, will use GEMINI_API_KEY environment variable)")
    parser.add_argument("--force_processing", action="store_true", help="Process all PDFs even if they would normally be filtered out")
    parser.add_argument("--max_pages_per_pdf", type=int, default=20, help="Maximum number of pages to process per PDF")
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
    processed_pdfs = 0
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        if process_pdf(pdf_path, args.output_dir, api_key, tests, args.max_pages_per_pdf, args.force_processing):
            processed_pdfs += 1

        # Save tests after each PDF to avoid losing data in case of crashes
        if tests:
            save_tests(tests, os.path.join(args.output_dir, "long_tests.jsonl"))

    print(f"Successfully processed {processed_pdfs} out of {len(pdf_files)} PDFs")
    print(f"Saved {len(tests)} tests to {os.path.join(args.output_dir, 'long_tests.jsonl')}")


if __name__ == "__main__":
    main()
