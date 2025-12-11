#!/usr/bin/env python3
"""
analyze_documents.py - Analyze document layout and extract content from PDF documents.

This script:
1. Takes a file containing S3 paths to PDF documents as input
2. For each PDF, extracts a random page and renders it to an image
3. Uses Gemini to analyze document layout features (columns, articles, text inserts, etc.)
4. If specific layout features are detected, proceeds with full document content extraction
5. Extracts the page from the PDF and saves it to an output folder along with analysis results

Usage:
  python analyze_documents.py --input_list path/to/s3_paths.txt --output_dir path/to/output --api_key your_gemini_api_key [--parallel 4]
"""

import argparse
import base64
import concurrent.futures
import json
import os
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pypdf
from google import genai
from google.genai import types
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.filter import PdfFilter

# Create a thread-safe lock for writing to output files
file_lock = threading.Lock()


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


def analyze_document_layout(pdf_path: str, page_num: int, api_key: str) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    Use Gemini to analyze document layout features in a rendered PDF page.

    Args:
        pdf_path: Path to the PDF file
        page_num: The page number to analyze (0-indexed)
        api_key: Gemini API key

    Returns:
        Optional[Tuple[Dict[str, Any], str]]:
            A tuple with the layout analysis results as a dictionary and the base64 string of the rendered page image.
            Returns None if analysis fails.
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

    # Prepare prompt for Gemini to analyze document layout
    contents = [
        types.Content(
            role="user",
            parts=[
                image_part,
                types.Part.from_text(
                    text=(
                        "Please answer the following questions about the document in JSON format:\n"
                        "-How many columns are used in the main text document layout?\n"
                        "-How many unique articles are captured in main text on this page?\n"
                        "-Are there any text inserts in the main article content?\n"
                        "-Do any of the main content articles start with a dropcap?\n"
                        "-Are there any boxed out regions of text that need to be read separately from the main article content?\n"
                        "-Are there any regions of text with a different orientation/rotation?"
                    )
                ),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.OBJECT,
            required=[
                "num_columns",
                "num_unique_articles",
                "contains_text_inserts",
                "contains_dropcaps",
                "contains_boxed_regions",
                "contains_text_different_orientation",
            ],
            properties={
                "num_columns": types.Schema(
                    type=types.Type.INTEGER,
                ),
                "num_unique_articles": types.Schema(
                    type=types.Type.INTEGER,
                ),
                "contains_text_inserts": types.Schema(
                    type=types.Type.BOOLEAN,
                ),
                "contains_dropcaps": types.Schema(
                    type=types.Type.BOOLEAN,
                ),
                "contains_boxed_regions": types.Schema(
                    type=types.Type.BOOLEAN,
                ),
                "contains_text_different_orientation": types.Schema(
                    type=types.Type.BOOLEAN,
                ),
            },
        ),
    )

    try:
        # Call Gemini API
        response = client.models.generate_content(model=model, contents=contents, config=generate_content_config)

        print(response)

        if not response.candidates or len(response.candidates) == 0:
            print(f"No response generated for {pdf_path} page {page_num}")
            return None

        if response.candidates[0].finish_reason != types.FinishReason.STOP:
            print(f"Response generation incomplete for {pdf_path} page {page_num}")
            return None

        # Parse the response
        response_text = response.candidates[0].content.parts[0].text

        layout_analysis = json.loads(response_text)

        print(f"Layout analysis for {pdf_path} page {page_num}:")
        print(json.dumps(layout_analysis, indent=2))

        # Return both the layout analysis and the rendered image (base64 string)
        return (layout_analysis, image_base64)

    except Exception as e:
        print(f"Error analyzing document layout in {pdf_path} page {page_num}: {str(e)}")
        return None


def extract_document_content(pdf_path: str, page_num: int, image_base64: str, api_key: str) -> Optional[str]:
    """
    Use Gemini to extract full document content from a rendered PDF page.

    Args:
        pdf_path: Path to the PDF file
        page_num: The page number to analyze (0-indexed)
        image_base64: The base64 string of the rendered page image
        api_key: Gemini API key

    Returns:
        Optional[str]: The extracted document content in markdown format, or None if extraction fails.
    """
    # Initialize Gemini client
    client = genai.Client(
        api_key=api_key,
    )
    model = "gemini-2.0-flash"

    image_part = types.Part(inline_data=types.Blob(mime_type="image/png", data=base64.b64decode(image_base64)))

    # Prepare prompt for Gemini to extract document content
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
            print(f"No response generated for content extraction in {pdf_path} page {page_num}")
            return None

        if response.candidates[0].finish_reason != types.FinishReason.STOP:
            print(f"Content extraction incomplete for {pdf_path} page {page_num}")
            return None

        # Get the extracted content
        content = response.candidates[0].content.parts[0].text
        return content

    except Exception as e:
        print(f"Error extracting document content from {pdf_path} page {page_num}: {str(e)}")
        return None


def should_extract_full_content(layout_analysis: Dict[str, Any]) -> bool:
    """
    Determine if full content extraction is needed based on layout analysis results.

    Args:
        layout_analysis: Dictionary containing layout analysis results

    Returns:
        bool: True if any of the special layout features are detected, False otherwise
    """
    # Check for special layout features that warrant full content extraction
    features_to_check = ["text_inserts", "dropcaps", "boxed_regions", "rotated_text"]

    # Also check if there are multiple columns or articles
    try:
        columns = layout_analysis.get("columns", 0)
        if isinstance(columns, str):
            columns = int(columns) if columns.isdigit() else 0

        articles = layout_analysis.get("articles", 0)
        if isinstance(articles, str):
            articles = int(articles) if articles.isdigit() else 0

        if columns > 1 or articles > 1:
            return True
    except (ValueError, TypeError):
        # If we can't parse the values, assume we need to extract
        pass

    # Check for any True values in the features
    for feature in features_to_check:
        value = layout_analysis.get(feature, False)
        if isinstance(value, str):
            if value.lower() in ["yes", "true", "1"]:
                return True
        elif value:
            return True

    return False


def process_pdf(s3_path: str, temp_dir: str, output_dir: str, api_key: str) -> Dict:
    """
    Process a single PDF from S3.

    Args:
        s3_path: S3 path to the PDF
        temp_dir: Directory for temporary files
        output_dir: Directory for output files
        api_key: Gemini API key

    Returns:
        Dict: Results of processing the PDF
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
        return {"error": f"Failed to download {s3_path}"}

    pdf_filter = PdfFilter()

    if pdf_filter.filter_out_pdf(local_pdf_path):
        print(f"Filtering out {pdf_filename}")
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
        return {"error": f"PDF {pdf_filename} filtered out"}

    try:
        # Read the PDF to get the number of pages
        reader = pypdf.PdfReader(local_pdf_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"PDF {pdf_filename} has no pages")
            return {"error": f"PDF {pdf_filename} has no pages"}

        all_pages = list(range(len(reader.pages)))
        random.shuffle(all_pages)

        results = {"filename": pdf_filename, "s3_path": s3_path}

        for page_num in all_pages:
            # Analyze document layout
            layout_result = analyze_document_layout(local_pdf_path, page_num, api_key)
            if not layout_result:
                print(f"Failed to analyze layout in {pdf_filename} page {page_num+1}")
                continue

            layout_analysis, image_base64 = layout_result
            results["layout_analysis"] = layout_analysis

            # Determine if we need to extract full content
            full_extraction_needed = should_extract_full_content(layout_analysis)
            results["full_extraction_needed"] = full_extraction_needed

            # Extract full content if needed
            if full_extraction_needed:
                content = extract_document_content(local_pdf_path, page_num, image_base64, api_key)
                results["content"] = content if content else "Content extraction failed"

            # Extract the page and save to output dir
            pdf_basename = os.path.splitext(pdf_filename)[0]
            output_pdf_path = os.path.join(output_dir, "pdfs", f"{pdf_basename}_pg{page_num+1}.pdf")
            with file_lock:  # Use lock when writing to shared output directory
                extract_page_from_pdf(local_pdf_path, output_pdf_path, page_num)

            # Save analysis results
            output_json_path = os.path.join(output_dir, "results", f"{pdf_basename}_pg{page_num+1}.json")
            with file_lock:
                os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
                with open(output_json_path, "w") as f:
                    json.dump(results, f, indent=2)

            print(f"Processed {pdf_filename} page {page_num+1}, analysis saved to {output_json_path}")

            # Process only one page per PDF
            break

        return results

    except Exception as e:
        print(f"Error processing {pdf_filename}: {str(e)}")
        return {"error": f"Error processing {pdf_filename}: {str(e)}"}
    finally:
        # Cleanup
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)


def process_pdfs_parallel(s3_paths: List[str], temp_dir: str, output_dir: str, api_key: str, max_docs: int, num_workers: int):
    """
    Process PDFs in parallel using a thread pool.

    Args:
        s3_paths: List of S3 paths to PDFs
        temp_dir: Directory for temporary files
        output_dir: Directory for output files
        api_key: Gemini API key
        max_docs: Maximum number of documents to process
        num_workers: Number of parallel workers to use
    """
    # Create output directory structure
    os.makedirs(os.path.join(output_dir, "pdfs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

    # Create a summary file
    summary_file = os.path.join(output_dir, "summary.jsonl")

    # Track processed documents
    processed_count = 0

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks and track futures
        futures = {executor.submit(process_pdf, s3_path, temp_dir, output_dir, api_key): s3_path for s3_path in s3_paths}

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            s3_path = futures[future]
            try:
                # Get the result from this worker
                result = future.result()

                # Add to summary file
                with file_lock:
                    with open(summary_file, "a") as f:
                        f.write(json.dumps(result) + "\n")

                # Increment counter if no error
                if "error" not in result:
                    processed_count += 1
                    print(f"Successfully processed {os.path.basename(s3_path)}, total: {processed_count}")

                # Check if we've reached the maximum number of documents
                if processed_count >= max_docs:
                    print(f"Reached maximum number of documents ({max_docs}), stopping")
                    # Cancel any pending futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break

            except Exception as e:
                print(f"Task for {os.path.basename(s3_path)} generated an exception: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze document layout and extract content from PDF documents")
    parser.add_argument("--input_list", required=True, help="Path to a file containing S3 paths to PDFs")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted pages and analysis results")
    parser.add_argument("--api_key", help="Gemini API key (if not provided, will use GEMINI_API_KEY environment variable)")
    parser.add_argument("--temp_dir", default="/tmp/analyze_documents", help="Directory for temporary files")
    parser.add_argument("--max_docs", type=int, default=100, help="Maximum number of documents to process")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel threads to use")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Gemini API key not provided. Use --api_key or set GEMINI_API_KEY environment variable.")
        return

    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pdfs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

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
    process_pdfs_parallel(s3_paths, args.temp_dir, args.output_dir, api_key, args.max_docs, num_workers)


if __name__ == "__main__":
    main()
