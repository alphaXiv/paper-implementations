#!/usr/bin/env python
import argparse
import json
import os
from collections import defaultdict

from pypdf import PdfReader, PdfWriter


def get_pdf_page_refs(dataset_jsonl):
    """
    Parse dataset.jsonl to extract all PDF page references.
    Returns a dict mapping (pdf_name, page_num) to a list of test IDs referencing that combination.
    """
    pdf_page_tests = defaultdict(list)

    with open(dataset_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                test = json.loads(line)
                pdf_name = test.get("pdf")
                page_num = test.get("page")
                test_id = test.get("id")

                if pdf_name and page_num and test_id:
                    pdf_page_tests[(pdf_name, page_num)].append(test_id)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line}")
                continue

    return pdf_page_tests


def extract_single_page_pdfs(source_pdf_dir, target_pdf_dir, pdf_page_tests):
    """
    Extract single page PDFs for each referenced (pdf_name, page_num) combination.
    """
    os.makedirs(target_pdf_dir, exist_ok=True)

    # Track all PDFs we need to process
    processed_pairs = set()

    for pdf_name, page_num in pdf_page_tests.keys():
        if (pdf_name, page_num) in processed_pairs:
            continue

        source_pdf_path = os.path.join(source_pdf_dir, pdf_name)
        if not os.path.exists(source_pdf_path):
            print(f"Warning: Source PDF not found: {source_pdf_path}")
            continue

        try:
            # Create a new single-page PDF
            reader = PdfReader(source_pdf_path)
            writer = PdfWriter()

            # PDF pages are 0-indexed, but our references are 1-indexed
            zero_indexed_page = page_num - 1

            if zero_indexed_page >= len(reader.pages) or zero_indexed_page < 0:
                print(f"Warning: Page {page_num} out of range for {pdf_name}")
                continue

            # Add the specified page to the writer
            writer.add_page(reader.pages[zero_indexed_page])

            # Create output filename
            # Remove .pdf extension if it exists
            base_name = pdf_name.rsplit(".", 1)[0] if pdf_name.lower().endswith(".pdf") else pdf_name
            output_filename = f"{base_name}_pg{page_num}.pdf"
            output_path = os.path.join(target_pdf_dir, output_filename)

            # Write the new PDF
            with open(output_path, "wb") as output_file:
                writer.write(output_file)

            processed_pairs.add((pdf_name, page_num))
            print(f"Created single-page PDF: {output_path}")

        except Exception as e:
            print(f"Error processing {pdf_name} page {page_num}: {str(e)}")

    return processed_pairs


def reorganize_test_outputs(source_data_dir, target_data_dir, processed_pairs):
    """
    Copy and reorganize test outputs matching the processed PDF page combinations.
    """
    # Create a dataset.jsonl with only the tests for pages we're keeping
    source_dataset = os.path.join(source_data_dir, "dataset.jsonl")
    target_dataset = os.path.join(target_data_dir, "dataset.jsonl")

    # Only copy tests for PDFs we processed
    if os.path.exists(source_dataset):
        with open(source_dataset, "r") as source_f, open(target_dataset, "w") as target_f:
            for line in source_f:
                if not line.strip():
                    continue

                try:
                    test = json.loads(line)
                    pdf_name = test.get("pdf")
                    page_num = test.get("page")

                    # Update the PDF name in the test to reflect our new naming convention
                    if (pdf_name, page_num) in processed_pairs:
                        base_name = pdf_name.rsplit(".", 1)[0] if pdf_name.lower().endswith(".pdf") else pdf_name
                        test["pdf"] = f"{base_name}_pg{page_num}.pdf"
                        # Since we've created single-page PDFs, update the page to 1
                        test["page"] = 1
                        target_f.write(json.dumps(test) + "\n")
                except json.JSONDecodeError:
                    continue


def main():
    parser = argparse.ArgumentParser(description="Extract single-page PDFs and reorganize test data")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory containing the original sample_data structure")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory to create the new data structure")
    args = parser.parse_args()

    source_data_dir = args.source_dir
    target_data_dir = args.target_dir

    # Create directory structure
    os.makedirs(target_data_dir, exist_ok=True)
    target_pdf_dir = os.path.join(target_data_dir, "pdfs")

    # Get paths
    source_dataset_path = os.path.join(source_data_dir, "dataset.jsonl")
    source_pdf_dir = os.path.join(source_data_dir, "pdfs")

    # Extract PDF page references from dataset
    pdf_page_tests = get_pdf_page_refs(source_dataset_path)
    print(f"Found {len(pdf_page_tests)} unique PDF page combinations referenced in tests")

    # Extract single-page PDFs
    processed_pairs = extract_single_page_pdfs(source_pdf_dir, target_pdf_dir, pdf_page_tests)
    print(f"Processed {len(processed_pairs)} unique PDF pages")

    # Reorganize test outputs
    reorganize_test_outputs(source_data_dir, target_data_dir, processed_pairs)

    print(f"Data extraction complete. New structure created in {target_data_dir}")


if __name__ == "__main__":
    main()
