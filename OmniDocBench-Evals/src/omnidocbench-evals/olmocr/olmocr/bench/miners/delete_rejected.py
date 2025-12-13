#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def get_rejected_tests(dataset_jsonl):
    """
    Parse dataset.jsonl to identify rejected tests.
    Returns:
    - rejected_tests: Set of test IDs that were marked as rejected
    - pdf_tests: Dict mapping PDF filenames to sets of test IDs
    - test_pdf_map: Dict mapping test IDs to their PDF filenames
    """
    rejected_tests = set()
    pdf_tests = defaultdict(set)
    test_pdf_map = {}

    try:
        with open(dataset_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    test = json.loads(line)
                    test_id = test.get("id")
                    pdf_name = test.get("pdf")

                    # Store the test in our mapping
                    if test_id and pdf_name:
                        pdf_tests[pdf_name].add(test_id)
                        test_pdf_map[test_id] = pdf_name

                    # Check if the test is marked as rejected
                    if test.get("checked", None) == "rejected":
                        rejected_tests.add(test_id)

                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                    continue

    except FileNotFoundError:
        print(f"Error: Dataset file {dataset_jsonl} not found.")
        sys.exit(1)

    return rejected_tests, pdf_tests, test_pdf_map


def update_dataset(dataset_jsonl, rejected_tests, dry_run=True):
    """
    Create a new dataset.jsonl without the rejected tests.
    """
    temp_file = dataset_jsonl + ".temp"
    removed_count = 0

    try:
        with open(dataset_jsonl, "r") as source, open(temp_file, "w") as target:
            for line in source:
                if not line.strip():
                    continue

                try:
                    test = json.loads(line)
                    test_id = test.get("id")

                    if test_id in rejected_tests:
                        removed_count += 1
                    else:
                        target.write(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Dataset file {dataset_jsonl} not found.")
        sys.exit(1)

    if not dry_run:
        os.replace(temp_file, dataset_jsonl)
    else:
        os.remove(temp_file)

    return removed_count


def find_orphaned_pdfs(pdf_dir, pdf_tests, rejected_tests):
    """
    Find PDF files that have all their tests rejected.
    """
    orphaned_pdfs = []

    for pdf_name, tests in pdf_tests.items():
        # Check if all tests for this PDF are in the rejected list
        if tests and all(test_id in rejected_tests for test_id in tests):
            pdf_path = os.path.join(pdf_dir, pdf_name)
            if os.path.exists(pdf_path):
                orphaned_pdfs.append(pdf_path)

    return orphaned_pdfs


def find_unreferenced_pdfs(pdf_dir, pdf_tests):
    """
    Find PDF files in the pdf_dir that are not referenced by any test.
    """
    unreferenced_pdfs = []
    # List all PDFs in the directory (recursively)
    for pdf_path in glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True):
        # Get the relative path of the PDF from pdf_dir
        pdf_name = os.path.relpath(pdf_path, pdf_dir)
        if pdf_name not in pdf_tests:
            unreferenced_pdfs.append(pdf_path)
    return unreferenced_pdfs


def main():
    parser = argparse.ArgumentParser(description="Delete rejected tests from dataset and orphaned/unreferenced PDFs")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset.jsonl files and the pdfs/ folder")
    parser.add_argument("--force", action="store_true", help="Perform actual deletion without confirmation")
    args = parser.parse_args()

    data_dir = args.data_dir
    dry_run = not args.force

    # Verify pdfs directory exists
    pdf_dir = os.path.join(data_dir, "pdfs")
    if not os.path.exists(pdf_dir):
        print(f"Error: pdfs/ directory not found in {data_dir}")
        sys.exit(1)

    # Find all JSONL dataset files in the data_dir
    dataset_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    if not dataset_files:
        print("No JSONL dataset files found.")
        sys.exit(0)

    # Global aggregation over all dataset files
    global_rejected_tests = set()
    global_pdf_tests = defaultdict(set)
    global_test_pdf_map = {}

    for dataset_file in dataset_files:
        rejected_tests, pdf_tests, test_pdf_map = get_rejected_tests(dataset_file)
        global_rejected_tests |= rejected_tests
        for pdf_name, test_ids in pdf_tests.items():
            global_pdf_tests[pdf_name].update(test_ids)
        global_test_pdf_map.update(test_pdf_map)

    total_tests = sum(len(test_ids) for test_ids in global_pdf_tests.values())

    # Compute orphaned and unreferenced PDFs using global mapping
    orphaned_pdfs = find_orphaned_pdfs(pdf_dir, global_pdf_tests, global_rejected_tests)
    unreferenced_pdfs = find_unreferenced_pdfs(pdf_dir, global_pdf_tests)

    # Print summary (global)
    print("\n===== DELETION SUMMARY =====")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'FORCE (changes will be applied)'}")
    print(f"Total tests: {total_tests}")
    print(f"Tests marked as rejected: {len(global_rejected_tests)}")
    print(f"PDF files with all tests rejected: {len(orphaned_pdfs)}")
    print(f"PDF files not referenced by any tests: {len(unreferenced_pdfs)}")

    if global_rejected_tests:
        print("\nRejected tests:")
        for test_id in sorted(global_rejected_tests):
            print(f"  - {test_id} (from {global_test_pdf_map.get(test_id, 'unknown')})")

    if orphaned_pdfs:
        print("\nPDF files to be deleted (all tests rejected):")
        for pdf_path in sorted(orphaned_pdfs):
            print(f"  - {os.path.basename(pdf_path)}")

    if unreferenced_pdfs:
        print("\nPDF files to be deleted (unreferenced by any tests):")
        for pdf_path in sorted(unreferenced_pdfs):
            print(f"  - {os.path.basename(pdf_path)}")

    # If dry run, exit here
    if dry_run and (global_rejected_tests or orphaned_pdfs or unreferenced_pdfs):
        print("\nThis is a dry run. No changes have been made.")
        print("To perform the actual deletion, run the script with the --force flag.")
        return

    # Confirm before deletion if there are items to delete
    if global_rejected_tests or orphaned_pdfs or unreferenced_pdfs:
        confirm = input("\nDo you want to proceed with deletion? (y/N): ")
        if confirm.lower() not in ("y", "yes"):
            print("Deletion cancelled.")
            return

        # Update each dataset file by removing rejected tests
        for dataset_file in dataset_files:
            removed_count = update_dataset(dataset_file, global_rejected_tests, dry_run=False)
            print(f"Removed {removed_count} rejected tests from {os.path.basename(dataset_file)}")

        # Delete orphaned PDFs
        for pdf_path in orphaned_pdfs:
            try:
                os.remove(pdf_path)
                print(f"Deleted orphaned PDF: {os.path.basename(pdf_path)}")
            except OSError as e:
                print(f"Error deleting {os.path.basename(pdf_path)}: {e}")

        # Delete unreferenced PDFs
        for pdf_path in unreferenced_pdfs:
            try:
                os.remove(pdf_path)
                print(f"Deleted unreferenced PDF: {os.path.basename(pdf_path)}")
            except OSError as e:
                print(f"Error deleting {os.path.basename(pdf_path)}: {e}")

        print("\nDeletion completed successfully.")
    else:
        print("\nNo rejected tests, orphaned PDFs, or unreferenced PDFs found. Nothing to delete.")


if __name__ == "__main__":
    main()
