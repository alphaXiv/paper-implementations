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

# Global state
DATASET_DIR = ""
DATASET_FILE = None
CURRENT_PDF = None
PDF_TESTS = {}
ALL_PDFS = []
FORCE = False


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
            return render_template("all_done_latex.html")

    # Get the tests for the current PDF
    current_tests = PDF_TESTS.get(CURRENT_PDF, [])

    # Create PDF URL for pdf.js to load
    pdf_url = url_for("serve_pdf", pdf_name=CURRENT_PDF)

    # Calculate statistics
    stats = calculate_stats()

    return render_template(
        "review_latex.html",
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

    # Create the review_latex.html template with MathJax support
    review_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <!-- You can adjust the viewport settings as needed -->
    <meta name="viewport" content="width=1200, initial-scale=1.0">
    <title>Equation Verification</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background-color: #f0f0f0;
            border-bottom: 1px solid #ddd;
        }
        .content {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .pdf-viewer {
            flex: 2; /* Increased from 1 to 2 to make PDF larger */
            border-right: 1px solid #ddd;
            overflow: hidden;
            position: relative;
        }
        /* Updated PDF container size */
        #pdf-container {
            width: 200%;  /* New fixed width */
            height: 200%;  /* New fixed height */
            overflow: auto;
        }
        #zoom-controls {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
            background-color: white;
            padding: 5px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        #zoom-controls button {
            margin: 0 5px;
            padding: 5px 10px;
            cursor: pointer;
        }
        .tests-panel {
            width: 1000px;
            overflow-y: auto;
            padding: 10px;
        }
        .test-item {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
            transition: background-color 0.2s;
        }
        .test-item.verified {
            background-color: #d4edda;
        }
        .test-item.rejected {
            background-color: #f8d7da;
        }
        /* The equation-display now stores the raw LaTeX in a data attribute */
        .equation-display {
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
            overflow-x: auto;
            font-size: 1.2em; /* Larger font for equations */
        }
        .button-group {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        }
        .button-group button {
            padding: 5px 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .verify-button {
            background-color: #28a745;
            color: white;
        }
        .reject-button {
            background-color: #dc3545;
            color: white;
        }
        .edit-button {
            background-color: #007bff;
            color: white;
        }
        .navigation {
            display: flex;
            align-items: center;
        }
        .status {
            margin-left: 20px;
        }
        .progress-bar {
            height: 10px;
            background-color: #e9ecef;
            border-radius: 5px;
            margin-top: 5px;
            overflow: hidden;
        }
        .progress {
            height: 100%;
            background-color: #007bff;
            width: 0%;
        }
        /* Make MathJax equations more visible */
        .MathJax {
            font-size: 120% !important;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h2>PDF: {{ pdf_name }}</h2>
            <form method="post" action="/reject_all" id="reject-all-form" style="display:inline;">
                <button type="button" onclick="rejectAll('{{ pdf_name }}')">Reject All Equations</button>
            </form>
        </div>
        <div class="navigation">
            <form method="post" action="/prev_pdf" style="display:inline;">
                <button type="submit" {% if pdf_index == 0 %}disabled{% endif %}>Previous</button>
            </form>
            <span style="margin: 0 10px;">{{ pdf_index + 1 }} / {{ total_pdfs }}</span>
            <form method="post" action="/next_pdf" style="display:inline;">
                <button type="submit">Next</button>
            </form>
            <div class="status">
                <div>Completion: {{ "%.1f"|format(stats.completion) }}%</div>
                <div class="progress-bar">
                    <div class="progress" style="width: {{ stats.completion }}%;"></div>
                </div>
            </div>
        </div>
    </div>
    <div class="content">
        <div class="pdf-viewer">
            <div id="zoom-controls">
                <button onclick="changeZoom(0.2)">+</button>
                <button onclick="changeZoom(-0.2)">-</button>
                <button onclick="resetZoom()">Reset</button>
            </div>
            <div id="pdf-container"></div>
        </div>
        <div class="tests-panel">
            <h3>Equations ({{ tests|length }})</h3>
            {% for test in tests %}
            <!-- Added data-latex attribute to store raw LaTeX -->
            <div class="test-item {% if test.checked == 'verified' %}verified{% elif test.checked == 'rejected' %}rejected{% endif %}" id="test-{{ test.id }}">
                <div class="equation-display" data-latex="{{ test.text|e }}">
                    {{ test.text|safe }}
                </div>
                <div class="button-group">
                    <button class="verify-button" onclick="updateTest('{{ test.id }}', '{{ test.pdf }}', 'checked', 'verified')">Verify</button>
                    <button class="reject-button" onclick="updateTest('{{ test.id }}', '{{ test.pdf }}', 'checked', 'rejected')">Reject</button>
                    <!-- New Edit button -->
                    <button class="edit-button" onclick="enableEdit('{{ test.id }}', '{{ test.pdf }}')">Edit</button>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        // Set up PDF.js
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';
        
        // Track current zoom level
        let currentScale = 2.0; // Initial larger scale
        let pdfDoc = null;
        let pageNum = 1;
        let canvas = null;
        
        // Load the PDF
        const loadingTask = pdfjsLib.getDocument('{{ pdf_path }}');
        loadingTask.promise.then(function(pdf) {
            pdfDoc = pdf;
            renderPage(pageNum);
        });
        
        // Function to render a page with the current scale
        function renderPage(num) {
            pdfDoc.getPage(num).then(function(page) {
                const viewport = page.getViewport({ scale: currentScale });
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    document.getElementById('pdf-container').appendChild(canvas);
                }
                const context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;
                const renderContext = {
                    canvasContext: context,
                    viewport: viewport
                };
                page.render(renderContext);
            });
        }
        
        // Function to change zoom level
        function changeZoom(delta) {
            currentScale += delta;
            if (currentScale < 0.5) currentScale = 0.5;
            if (currentScale > 5) currentScale = 5;
            renderPage(pageNum);
        }
        
        // Function to reset zoom
        function resetZoom() {
            currentScale = 2.0;
            renderPage(pageNum);
        }
        
        // Function to update a test – used by both verify/reject and edit
        function updateTest(testId, pdfName, field, value) {
            fetch('/update_test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    id: testId,
                    pdf: pdfName,
                    field: field,
                    value: value
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const testElement = document.getElementById(`test-${testId}`);
                    testElement.classList.remove('verified', 'rejected');
                    // Only update the class if the field updated is "checked".
                    if (field === 'checked') {
                        testElement.classList.add(value);
                    }
                }
            });
        }
        
        // New function to enable editing the LaTeX equation
        function enableEdit(testId, pdfName) {
            let testElement = document.getElementById("test-" + testId);
            let equationDisplay = testElement.querySelector(".equation-display");
            // Retrieve the raw LaTeX from the data attribute
            let rawLatex = equationDisplay.getAttribute('data-latex');
            // Save the current rendered HTML in case of cancellation
            let originalHTML = equationDisplay.innerHTML;
            
            // Create a textarea for editing the LaTeX
            let textarea = document.createElement("textarea");
            textarea.id = "edit-input-" + testId;
            // Use the stored raw LaTeX if available; otherwise, fallback to textContent
            textarea.value = rawLatex ? rawLatex : equationDisplay.textContent.trim();
            textarea.style.width = "100%";
            textarea.rows = 3;
            
            // Create Save button to commit changes
            let saveButton = document.createElement("button");
            saveButton.innerText = "Save";
            saveButton.onclick = function() {
                let newText = textarea.value;
                // Update the test via AJAX – updating the 'text' field
                updateTest(testId, pdfName, "text", newText);
                // Update the data-latex attribute to hold the new raw LaTeX code
                equationDisplay.setAttribute('data-latex', newText);
                // Replace the display content with the wrapped LaTeX for MathJax to process
                equationDisplay.innerHTML = '$$' + newText + '$$';
                if (typeof MathJax !== 'undefined') {
                    MathJax.typeset();
                }
                // Clean up the temporary editing elements
                textarea.remove();
                saveButton.remove();
                cancelButton.remove();
            };
            
            // Create Cancel button to revert changes
            let cancelButton = document.createElement("button");
            cancelButton.innerText = "Cancel";
            cancelButton.onclick = function() {
                equationDisplay.innerHTML = originalHTML;
                textarea.remove();
                saveButton.remove();
                cancelButton.remove();
            };
            
            // Show the editing interface: clear the display and insert the textarea
            equationDisplay.innerHTML = "";
            equationDisplay.appendChild(textarea);
            // Append buttons to the test element (or you can choose to append them elsewhere)
            testElement.appendChild(saveButton);
            testElement.appendChild(cancelButton);
        }
        
        // Function to reject all tests for a PDF
        function rejectAll(pdfName) {
            if (confirm('Are you sure you want to reject all equations for this PDF?')) {
                fetch('/reject_all', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        pdf: pdfName
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        const testElements = document.querySelectorAll('.test-item');
                        testElements.forEach(element => {
                            element.classList.remove('verified');
                            element.classList.add('rejected');
                        });
                    }
                });
            }
        }

        // Process LaTeX equations on page load: wrap plain text with $$ and trigger MathJax typesetting
        document.addEventListener('DOMContentLoaded', function() {
            const equationDisplays = document.querySelectorAll('.equation-display');
            equationDisplays.forEach(display => {
                let equation = display.textContent.trim();
                if (!equation.startsWith('$$')) {
                    display.innerHTML = '$$' + equation + '$$';
                }
            });
            if (typeof MathJax !== 'undefined') {
                MathJax.typeset();
            }
        });
    </script>
</body>
</html>
    """

    # Create the all_done_latex.html template
    all_done_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All Done!</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f5f5f5;
        }
        .container {
            text-align: center;
            padding: 30px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #28a745;
            margin-bottom: 20px;
        }
        p {
            font-size: 18px;
            margin-bottom: 20px;
        }
        button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>All Done! 🎉</h1>
        <p>You have reviewed all equations in the dataset.</p>
        <form method="post" action="/next_pdf">
            <button type="submit">Start Over</button>
        </form>
    </div>
</body>
</html>
    """

    with open(os.path.join(templates_dir, "review_latex.html"), "w") as f:
        f.write(review_html)

    with open(os.path.join(templates_dir, "all_done_latex.html"), "w") as f:
        f.write(all_done_html)


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
    FORCE = args.force

    if not os.path.exists(args.dataset_file):
        print(f"Error: Dataset not found: {args.dataset_file}")
        return 1

    DATASET_DIR = os.path.dirname(os.path.abspath(args.dataset_file))
    DATASET_FILE = args.dataset_file

    pdf_dir = os.path.join(DATASET_DIR, "pdfs")
    if not os.path.isdir(pdf_dir):
        print(f"Error: PDF directory not found: {pdf_dir}")
        return 1

    try:
        PDF_TESTS, ALL_PDFS = load_dataset(args.dataset_file)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return 1

    create_templates_directory()
    CURRENT_PDF = find_next_unchecked_pdf()

    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
