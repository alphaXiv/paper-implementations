#!/usr/bin/env python3
"""
mine_tables.py - Extract tables from PDF documents and create table tests.

This script:
1. Takes a file containing S3 paths to PDF documents as input
2. For each PDF, extracts a random page and renders it to an image
3. Uses Gemini to identify tables in the rendered image
4. Extracts table content and creates table relationship tests by making a second Gemini request
   that now includes the page image alongside the prompt (e.g., "Given cell with {cell_value}, which cell is directly to the left of it?")
5. Extracts the page from the PDF and saves it to an output folder

Usage:
  python mine_tables.py --input_list path/to/s3_paths.txt --output_dir path/to/output --api_key your_gemini_api_key [--parallel 4]
"""

import argparse
import base64
import concurrent.futures
import os
import random
import threading
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pypdf
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from tqdm import tqdm

from olmocr.bench.tests import TableTest, save_tests
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.filter import PdfFilter

# Create a thread-safe lock for writing to the output file
file_lock = threading.Lock()
tests_lock = threading.Lock()


def download_pdf_from_s3(s3_path: str, local_path: str) -> bool:
    """
    Download a PDF file from S3.

    Args:
        s3_path: The S3 path (s3://bucket/path/to/file.pdf)
        local_path: The local path to save the file

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Parse S3 path
        parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]

        # Create S3 client
        s3 = boto3.client("s3")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download file
        s3.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {str(e)}")
        return False


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


def detect_tables(pdf_path: str, page_num: int, api_key: str) -> Optional[Tuple[List[np.ndarray], str]]:
    """
    Use Gemini to detect tables in a rendered PDF page.

    Args:
        pdf_path: Path to the PDF file
        page_num: The page number to analyze (0-indexed)
        api_key: Gemini API key

    Returns:
        Optional[Tuple[List[np.ndarray], str]]:
            A tuple with a list of detected tables (as numpy arrays) and the base64 string of the rendered page image.
            Returns None if detection fails.
    """
    # Initialize Gemini client
    client = genai.Client(
        api_key=api_key,
    )
    model = "gemini-2.0-flash"

    # Render the PDF page as an image (render_pdf_to_base64png is 1-indexed)
    try:
        image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num + 1, target_longest_image_dim=2048)
    except Exception as e:
        print(f"Error rendering PDF page: {str(e)}")
        return None

    image_part = types.Part(inline_data=types.Blob(mime_type="image/png", data=base64.b64decode(image_base64)))

    # Prepare prompt for Gemini to extract tables
    contents = [
        types.Content(
            role="user",
            parts=[
                image_part,
                types.Part.from_text(
                    text=(
                        "Analyze the document attached and output it in markdown format. "
                        "Output equations as Latex escaped with $$. "
                        "Output tables in HTML format that preserves the structure and content exactly, do not use <br> tags. "
                        "Instead of the markdown table format, be sure to output tables in HTML, even though the rest of the document is styled in markdown. "
                        "Output figures with just a simple markdown image placeholder."
                    )
                ),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(temperature=0.2, top_p=0.95, top_k=40, max_output_tokens=8192)

    try:
        # Call Gemini API
        response = client.models.generate_content(model=model, contents=contents, config=generate_content_config)

        if not response.candidates or len(response.candidates) == 0:
            print(f"No response generated for {pdf_path} page {page_num}")
            return None

        if response.candidates[0].finish_reason != types.FinishReason.STOP:
            print(f"Response generation incomplete for {pdf_path} page {page_num}")
            return None

        # Parse the response
        response_text = response.candidates[0].content.parts[0].text

        print(response_text)

        # Parse tables from HTML
        parsed_tables = []
        soup = BeautifulSoup(response_text, "html.parser")
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")
            table_data = []
            for row in rows:
                cells = row.find_all(["th", "td"])
                row_data = [cell.get_text().strip() for cell in cells]
                table_data.append(row_data)
            # Ensure all rows have the same number of columns
            if table_data:
                max_cols = max(len(row) for row in table_data)
                padded_data = [row + [""] * (max_cols - len(row)) for row in table_data]
                table_array = np.array(padded_data)
                parsed_tables.append(table_array)

        # Return both the parsed tables and the rendered image (base64 string)
        return (parsed_tables, image_base64) if parsed_tables else None

    except Exception as e:
        print(f"Error detecting tables in {pdf_path} page {page_num}: {str(e)}")
        return None


def generate_table_tests(tables: List[np.ndarray], pdf_image: str, api_key: str, max_tests_per_table: int = 3) -> List[Dict]:
    """
    Generate table tests from the detected tables by making a second Gemini request for each candidate cell.

    For each candidate cell in a table, the function selects one valid relationship (e.g., "left", "up", "top_heading", etc.)
    and sends a prompt to Gemini including the page image. For example:
      "Given a cell in a table with value 'XYZ', please answer: which cell is directly to the left of it? Provide only the cell's text."

    Args:
        tables: List of tables as numpy arrays
        pdf_image: Base64 string of the rendered page image
        api_key: Gemini API key to use for generating relationship tests
        max_tests_per_table: Maximum number of tests to generate per table

    Returns:
        List of table test dictionaries
    """
    tests = []
    # Initialize Gemini client for test queries
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"
    config = types.GenerateContentConfig(temperature=0.2, top_p=0.95, top_k=40, max_output_tokens=100)

    # Mapping for relationship prompts
    prompt_map = {
        "up": "which cell is directly above it?",
        "down": "which cell is directly below it?",
        "left": "which cell is directly to the left of it?",
        "right": "which cell is directly to the right of it?",
        "top_heading": "what is the top heading (the heading for the column at the top of the table) for this cell?",
        "left_heading": "what is the left heading (the heading for this row on the left part of the table) for this cell?",
    }

    # Create an image part from the rendered pdf image
    image_part = types.Part(inline_data=types.Blob(mime_type="image/png", data=base64.b64decode(pdf_image)))

    for table in tables:
        rows, cols = table.shape
        if table.size == 0 or rows < 2 or cols < 2:
            continue  # Skip tables that are too small

        # Try up to 3x max_tests_per_table candidate cells
        candidate_positions = []
        for row in range(rows):
            for col in range(cols):
                if not table[row, col].strip():
                    continue
                if row > 0:
                    candidate_positions.append((row, col, "up"))
                if row < rows - 1:
                    candidate_positions.append((row, col, "down"))
                if col > 0:
                    candidate_positions.append((row, col, "left"))
                if col < cols - 1:
                    candidate_positions.append((row, col, "right"))
                if row > 0:
                    candidate_positions.append((row, col, "top_heading"))
                if col > 0:
                    candidate_positions.append((row, col, "left_heading"))

        random.shuffle(candidate_positions)
        tests_for_this_table = 0

        for row, col, relationship in candidate_positions:
            if tests_for_this_table >= max_tests_per_table:
                break

            cell_value = table[row, col].strip()

            prompt = (
                f"Given a cell in a table with value '{cell_value}', please answer: "
                f"{prompt_map[relationship]} Provide only the cell's text or output 'null' if there is not a matching cell."
            )

            try:
                contents = [types.Content(role="user", parts=[image_part, types.Part.from_text(text=prompt)])]
                response = client.models.generate_content(model=model, contents=contents, config=config)
                if not response.candidates or len(response.candidates) == 0 or response.candidates[0].finish_reason != types.FinishReason.STOP:
                    continue
                answer_text = response.candidates[0].content.parts[0].text.strip()
                if answer_text and "null" not in answer_text:
                    test_data = {"cell": cell_value, relationship: answer_text}
                    tests.append(test_data)
                    tests_for_this_table += 1
            except Exception as e:
                print(f"Error querying Gemini for cell '{cell_value}' and relationship '{relationship}': {str(e)}")

    return tests


def process_pdf(s3_path: str, temp_dir: str, output_dir: str, api_key: str) -> List[TableTest]:
    """
    Process a single PDF from S3.

    Args:
        s3_path: S3 path to the PDF
        temp_dir: Directory for temporary files
        output_dir: Directory for output files
        api_key: Gemini API key

    Returns:
        List[TableTest]: List of generated table tests
    """
    # Create a thread-specific temp directory to avoid conflicts
    thread_id = threading.get_ident()
    thread_temp_dir = os.path.join(temp_dir, f"thread_{thread_id}")
    os.makedirs(thread_temp_dir, exist_ok=True)

    # Extract filename from S3 path
    pdf_filename = os.path.basename(s3_path)
    local_pdf_path = os.path.join(thread_temp_dir, pdf_filename)

    # Download PDF from S3
    if not download_pdf_from_s3(s3_path, local_pdf_path):
        return []

    pdf_filter = PdfFilter()

    if pdf_filter.filter_out_pdf(local_pdf_path):
        print(f"Filtering out {pdf_filename}")
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
        return []

    try:
        # Read the PDF to get the number of pages
        reader = pypdf.PdfReader(local_pdf_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"PDF {pdf_filename} has no pages")
            return []

        all_pages = list(range(len(reader.pages)))
        random.shuffle(all_pages)

        local_tests = []

        for page_num in all_pages:
            # Detect tables and obtain the rendered image for this page
            result = detect_tables(local_pdf_path, page_num, api_key)
            if not result:
                print(f"No tables detected in {pdf_filename} page {page_num+1}")
                continue

            tables, image_base64 = result

            # Generate table tests using the new Gemini query approach with the page image
            table_tests_data = generate_table_tests(tables, image_base64, api_key, max_tests_per_table=5)

            if not table_tests_data:
                print(f"Could not generate valid tests for tables in {pdf_filename} page {page_num+1}")
                continue

            # Extract the page and save to output dir
            pdf_basename = os.path.splitext(pdf_filename)[0]
            output_pdf_path = os.path.join(output_dir, "pdfs", f"{pdf_basename}_pg{page_num+1}.pdf")
            with file_lock:  # Use lock when writing to shared output directory
                extract_page_from_pdf(local_pdf_path, output_pdf_path, page_num)

            # Create table tests
            for i, test_data in enumerate(table_tests_data):
                test_id = f"{pdf_basename}_pg{page_num+1}_table_{i:02d}"
                test = TableTest(
                    id=test_id,
                    pdf=f"{pdf_basename}_pg{page_num+1}.pdf",
                    page=1,  # The extracted PDF has only one page
                    type="table",
                    cell=test_data["cell"],
                    url=s3_path,  # Added the S3 path as the url field
                    up=test_data.get("up", None),
                    down=test_data.get("down", None),
                    left=test_data.get("left", None),
                    right=test_data.get("right", None),
                    top_heading=test_data.get("top_heading", None),
                    left_heading=test_data.get("left_heading", None),
                )
                local_tests.append(test)

            print(f"Processed {pdf_filename} page {page_num+1}, found {len(tables)} tables, created {len(table_tests_data)} tests")
            break  # Process only one page per PDF

        return local_tests

    except Exception as e:
        print(f"Error processing {pdf_filename}: {str(e)}")
        return []
    finally:
        # Cleanup
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)


def process_pdfs_parallel(s3_paths: List[str], temp_dir: str, output_dir: str, api_key: str, max_tests: int, num_workers: int):
    """
    Process PDFs in parallel using a thread pool.

    Args:
        s3_paths: List of S3 paths to PDFs
        temp_dir: Directory for temporary files
        output_dir: Directory for output files
        api_key: Gemini API key
        max_tests: Maximum number of tests to generate
        num_workers: Number of parallel workers to use
    """
    # Create shared resources
    all_tests = []
    output_file = os.path.join(output_dir, "table_tests.jsonl")

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks and track futures
        futures = {executor.submit(process_pdf, s3_path, temp_dir, output_dir, api_key): s3_path for s3_path in s3_paths}

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            s3_path = futures[future]
            try:
                # Get the tests produced by this worker
                new_tests = future.result()

                # If we got new tests, add them to our collection
                if new_tests:
                    all_tests.extend(new_tests)
                    save_tests(all_tests, output_file)
                    print(f"Added {len(new_tests)} tests from {os.path.basename(s3_path)}, total: {len(all_tests)}")

                    # Check if we've reached the maximum number of tests
                    if len(all_tests) >= max_tests:
                        print(f"Reached maximum number of tests ({max_tests}), stopping")
                        # Cancel any pending futures
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break

            except Exception as e:
                print(f"Task for {os.path.basename(s3_path)} generated an exception: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract tables from PDF documents and create table tests")
    parser.add_argument("--input_list", required=True, help="Path to a file containing S3 paths to PDFs")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted pages and tests")
    parser.add_argument("--api_key", help="Gemini API key (if not provided, will use GEMINI_API_KEY environment variable)")
    parser.add_argument("--temp_dir", default="/tmp/mine_tables", help="Directory for temporary files")
    parser.add_argument("--max_tests", type=int, default=100, help="Maximum number of tests to generate")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel threads to use")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Gemini API key not provided. Use --api_key or set GEMINI_API_KEY environment variable.")
        return

    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pdfs"), exist_ok=True)

    # Reservoir sampling implementation
    s3_paths = []
    with open(args.input_list, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if not line:
                continue

            if i < 100000:
                s3_paths.append(line)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, i)
                if j < 100000:
                    s3_paths[j] = line

    print(f"Found {len(s3_paths)} PDF paths in input list")

    # Determine number of workers to use
    num_workers = max(1, min(args.parallel, len(s3_paths)))
    print(f"Processing PDFs using {num_workers} parallel workers")

    # Process PDFs in parallel
    process_pdfs_parallel(s3_paths, args.temp_dir, args.output_dir, api_key, args.max_tests, num_workers)


if __name__ == "__main__":
    main()
