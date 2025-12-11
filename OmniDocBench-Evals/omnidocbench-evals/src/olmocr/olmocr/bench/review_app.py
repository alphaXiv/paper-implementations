#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

app = Flask(__name__)
# Add static folder for KaTeX files
app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "katex")

# Global state
DATASET_DIR = ""
DATASET_FILE = None
CURRENT_PDF = None
PDF_TESTS = {}
ALL_PDFS = []
FORCE = False  # New global flag


def find_next_unchecked_pdf() -> Optional[str]:
    """Find the next PDF with at least one unchecked test."""
    global PDF_TESTS, ALL_PDFS

    for pdf_name in ALL_PDFS:
        pdf_tests = PDF_TESTS[pdf_name]
        for test in pdf_tests:
            if test.get("checked") is None:
                return pdf_name
    return None


def calculate_stats() -> dict:
    """Calculate statistics for all tests in the dataset."""
    global PDF_TESTS

    total_tests = 0
    null_status = 0
    verified_status = 0
    rejected_status = 0

    for pdf_tests in PDF_TESTS.values():
        total_tests += len(pdf_tests)

        for test in pdf_tests:
            status = test.get("checked")
            if status is None:
                null_status += 1
            elif status == "verified":
                verified_status += 1
            elif status == "rejected":
                rejected_status += 1

    completion = 0
    if total_tests > 0:
        completion = (verified_status + rejected_status) / total_tests * 100

    return {"total": total_tests, "null": null_status, "verified": verified_status, "rejected": rejected_status, "completion": completion}


def save_dataset(jsonl_file: str) -> None:
    """Save the tests to a JSONL file, using temp file for atomic write."""
    global PDF_TESTS

    # Flatten all tests
    all_tests = []
    for pdf_tests in PDF_TESTS.values():
        all_tests.extend(pdf_tests)

    # Create temp file and write updated content
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        for test in all_tests:
            temp_file.write(json.dumps(test) + "\n")

    # Atomic replace
    shutil.move(temp_file.name, jsonl_file)


@app.route("/pdf/<path:pdf_name>")
def serve_pdf(pdf_name):
    """Serve the PDF file directly."""
    pdf_path = os.path.join(DATASET_DIR, "pdfs", pdf_name)
    return send_file(pdf_path, mimetype="application/pdf")


@app.route("/")
def index():
    """Main page displaying the current PDF and its tests."""
    global CURRENT_PDF, PDF_TESTS, DATASET_DIR, ALL_PDFS, FORCE

    # If no current PDF is set, find the next one with unchecked tests
    if CURRENT_PDF is None:
        CURRENT_PDF = find_next_unchecked_pdf()

    # If still no PDF, either show the "All done" page or force display the first PDF
    if CURRENT_PDF is None:
        if FORCE and ALL_PDFS:
            CURRENT_PDF = ALL_PDFS[0]
        else:
            return render_template("all_done.html")

    # Get the tests for the current PDF
    current_tests = PDF_TESTS.get(CURRENT_PDF, [])

    # Create PDF URL for pdf.js to load
    pdf_url = url_for("serve_pdf", pdf_name=CURRENT_PDF)

    # Calculate statistics
    stats = calculate_stats()

    return render_template(
        "review.html",
        pdf_name=CURRENT_PDF,
        tests=current_tests,
        pdf_path=pdf_url,
        pdf_index=ALL_PDFS.index(CURRENT_PDF) if CURRENT_PDF in ALL_PDFS else 0,
        total_pdfs=len(ALL_PDFS),
        stats=stats,
    )


@app.route("/update_test", methods=["POST"])
def update_test():
    """API endpoint to update a test."""
    global PDF_TESTS, DATASET_DIR, DATASET_FILE

    data = request.json
    pdf_name = data.get("pdf")
    test_id = data.get("id")
    field = data.get("field")
    value = data.get("value")

    # Find and update the test
    for test in PDF_TESTS.get(pdf_name, []):
        if test.get("id") == test_id:
            test[field] = value
            break

    # Save the updated tests
    save_dataset(DATASET_FILE)

    return jsonify({"status": "success"})


@app.route("/reject_all", methods=["POST"])
def reject_all():
    """API endpoint to reject all tests for a PDF."""
    global PDF_TESTS, DATASET_DIR, DATASET_FILE

    data = request.json
    pdf_name = data.get("pdf")

    if pdf_name and pdf_name in PDF_TESTS:
        # Update all tests for this PDF to rejected
        for test in PDF_TESTS[pdf_name]:
            test["checked"] = "rejected"

        # Save the updated tests
        save_dataset(DATASET_FILE)

        return jsonify({"status": "success", "count": len(PDF_TESTS[pdf_name])})

    return jsonify({"status": "error", "message": "PDF not found"})


@app.route("/next_pdf", methods=["POST"])
def next_pdf():
    """Move to the next PDF in the list."""
    global CURRENT_PDF, ALL_PDFS, FORCE

    if CURRENT_PDF in ALL_PDFS:
        current_index = ALL_PDFS.index(CURRENT_PDF)
        if current_index < len(ALL_PDFS) - 1:
            CURRENT_PDF = ALL_PDFS[current_index + 1]
        else:
            # If in force mode, cycle back to the beginning instead of checking for an unchecked PDF
            if FORCE and ALL_PDFS:
                CURRENT_PDF = ALL_PDFS[0]
            else:
                CURRENT_PDF = find_next_unchecked_pdf()
    else:
        if FORCE and ALL_PDFS:
            CURRENT_PDF = ALL_PDFS[0]
        else:
            CURRENT_PDF = find_next_unchecked_pdf()

    return redirect(url_for("index"))


@app.route("/prev_pdf", methods=["POST"])
def prev_pdf():
    """Move to the previous PDF in the list."""
    global CURRENT_PDF, ALL_PDFS

    if CURRENT_PDF in ALL_PDFS:
        current_index = ALL_PDFS.index(CURRENT_PDF)
        if current_index > 0:
            CURRENT_PDF = ALL_PDFS[current_index - 1]

    return redirect(url_for("index"))


@app.route("/goto_pdf/<int:index>", methods=["POST"])
def goto_pdf(index):
    """Go to a specific PDF by index."""
    global CURRENT_PDF, ALL_PDFS

    if 0 <= index < len(ALL_PDFS):
        CURRENT_PDF = ALL_PDFS[index]

    return redirect(url_for("index"))


def load_dataset(dataset_file: str) -> Tuple[Dict[str, List[Dict]], List[str]]:
    """Load tests from the dataset file and organize them by PDF."""
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    pdf_tests = defaultdict(list)

    with open(dataset_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                test = json.loads(line)
                pdf_name = test.get("pdf")
                if pdf_name:
                    pdf_tests[pdf_name].append(test)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON: {line}")

    all_pdfs = list(pdf_tests.keys())

    return pdf_tests, all_pdfs


def create_templates_directory():
    """Create templates directory for Flask if it doesn't exist."""
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)


def main():
    """Main entry point with command-line arguments."""
    global DATASET_DIR, DATASET_FILE, PDF_TESTS, ALL_PDFS, CURRENT_PDF, FORCE

    parser = argparse.ArgumentParser(description="Interactive Test Review App")
    parser.add_argument("dataset_file", help="Path to the dataset jsonl file")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Flask app")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the Flask app")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    parser.add_argument("--force", action="store_true", help="Force show each file one by one and never do the 'All done' page")

    args = parser.parse_args()
    FORCE = args.force  # Set the global FORCE flag

    # Validate dataset directory
    if not os.path.exists(args.dataset_file):
        print(f"Error: Dataset not found: {args.dataset_file}")
        return 1

    # Store dataset directory globally
    DATASET_DIR = os.path.dirname(os.path.abspath(args.dataset_file))
    DATASET_FILE = args.dataset_file

    pdf_dir = os.path.join(DATASET_DIR, "pdfs")
    if not os.path.isdir(pdf_dir):
        print(f"Error: PDF directory not found: {pdf_dir}")
        return 1

    # Load dataset
    try:
        PDF_TESTS, ALL_PDFS = load_dataset(args.dataset_file)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return 1

    # Create templates directory
    create_templates_directory()

    # Find first PDF with unchecked tests
    CURRENT_PDF = find_next_unchecked_pdf()

    # Start Flask app
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
