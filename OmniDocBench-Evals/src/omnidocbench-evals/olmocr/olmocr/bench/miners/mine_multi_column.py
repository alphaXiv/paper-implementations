"""
mine_multi_column.py - Extract text from PDF documents which has multiple columns.

This script:
1. Takes a file containing folder path which contains PDF documents as input
2. Process each PDF to generate an HTML representation
3. For each PDF, it renders to an image
4. Uses Claude Sonnet to identify text from multiple columns in the rendered image
5. Creates a test file asserting that the order (before/after) of text should appear
6. Extracts the page from the PDF and saves it to an output folder

Usage:
  python mine_headers_footers.py --input_dir path/to/pdfs --output_dir path/to/output --api_key your_anthropic_api_key
"""

import argparse
import asyncio
import concurrent.futures
import json
import os
import random
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pypdf
from anthropic import Anthropic
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from syntok.segmenter import process
from tqdm import tqdm

from olmocr.data.renderpdf import (
    get_png_dimensions_from_base64,
    render_pdf_to_base64png,
)


def extract_code_block(initial_response):
    html_blocks = re.findall(r"```html\n(.*?)```", initial_response, re.DOTALL)
    if html_blocks:
        return html_blocks[-1].strip()
    code_blocks = re.findall(r"```\n(.*?)```", initial_response, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    html_blocks_no_newline = re.findall(r"```html(.*?)```", initial_response, re.DOTALL)
    if html_blocks_no_newline:
        return html_blocks_no_newline[-1].strip()
    code_blocks_no_newline = re.findall(r"```(.*?)```", initial_response, re.DOTALL)
    if code_blocks_no_newline:
        return code_blocks_no_newline[-1].strip()
    return None


def generate_html_from_image(client, image_base64):
    """Call Claude API to generate HTML from an image using a multi-step prompting strategy."""
    png_width, png_height = get_png_dimensions_from_base64(image_base64)
    try:
        analysis_response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {
                            "type": "text",
                            "text": (
                                "Analyze this document and provide a detailed assessment of its structure. "
                                "Focus on the layout, headings, footers, and any complex formatting. Please be precise."
                            ),
                        },
                    ],
                }
            ],
        )

        analysis_text = ""
        for content in analysis_response.content:
            if content.type == "text":
                analysis_text += content.text

        initial_response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=6000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {
                            "type": "text",
                            "text": (
                                "Render this document as clean, semantic HTML. Here is the analysis of the document structure:\n\n"
                                f"{analysis_text}\n\n"
                                "Requirements:\n"
                                "1. Use appropriate HTML tags for headings, paragraphs, and lists.\n"
                                "2. Use <header> and <footer> for top and bottom content.\n"
                                "3. For images, use a placeholder <div> with class 'image'.\n"
                                "4. Render math equations inline using \\( \\) or \\[ \\].\n"
                                "5. Preserve any multi-column layout using CSS flexbox or grid.\n"
                                f"6. The viewport is fixed at {png_width // 2}x{png_height // 2} pixels.\n\n"
                                "Enclose your HTML in a ```html code block."
                            ),
                        },
                    ],
                }
            ],
        )

        initial_html = ""
        for content in initial_response.content:
            if content.type == "text":
                initial_html += content.text

        return extract_code_block(initial_html)
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None


def generate_tests_from_html(html_content: str, pdf_id: str, page_num: int) -> List[Dict]:
    """
    Generate order tests from HTML content by splitting the main text into sentences and pairing them.
    Only order tests are generated.
    """
    tests = []
    pdf_filename = pdf_id
    soup = BeautifulSoup(html_content, "html.parser")
    full_text = soup.get_text(separator=" ").strip()

    sentences = []
    for paragraph in process(full_text):
        for sentence in paragraph:
            sentence_str = "".join(token.spacing + token.value for token in sentence).strip()
            if sentence_str:
                sentences.append(sentence_str)

    if len(sentences) < 2:
        return tests

    all_indexes = list(range(len(sentences)))
    random.shuffle(all_indexes)
    random_pairs = [(all_indexes[i * 2], all_indexes[i * 2 + 1]) for i in range(len(all_indexes) // 2)]
    random_pairs = [(min(i, j), max(i, j)) for (i, j) in random_pairs]

    num_order_tests = 0
    for i, j in random_pairs:
        first_sentence = sentences[i]
        second_sentence = sentences[j]
        if len(first_sentence) < 10 or len(second_sentence) < 10:
            continue
        first_sentence = first_sentence.split("\n")[0].strip()
        second_sentence = second_sentence.split("\n")[0].strip()
        max_diffs = round(max(len(first_sentence), len(second_sentence)) * 0.05)
        if max_diffs > len(first_sentence) // 2 or max_diffs > len(second_sentence) // 2:
            continue
        tests.append(
            {
                "pdf": pdf_filename,
                "page": page_num,
                "id": f"{pdf_id}_order_{uuid.uuid4().hex[:8]}",
                "type": "order",
                "before": first_sentence,
                "after": second_sentence,
                "max_diffs": max_diffs,
            }
        )
        num_order_tests += 1
        if num_order_tests >= 5:
            break

    return tests


async def render_pdf_with_playwright(html_content, output_pdf_path, png_width, png_height):
    """
    Render HTML content using Playwright and save it as a PDF.
    Tries different scale factors until the output PDF has exactly one page.
    """
    scale_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for scale in scale_factors:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page(viewport={"width": int(png_width // 2 * scale), "height": int(png_height // 2 * scale)})
                await page.set_content(html_content)

                # Add KaTeX assets for math rendering
                katex_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "katex")
                katex_css_path = os.path.join(katex_dir, "katex.min.css")
                katex_js_path = os.path.join(katex_dir, "katex.min.js")
                katex_autorender_js_path = os.path.join(katex_dir, "auto-render.min.js")

                await page.add_style_tag(path=katex_css_path)
                await page.add_script_tag(path=katex_js_path)
                await page.add_script_tag(path=katex_autorender_js_path)

                await page.evaluate(
                    """
                    renderMathInElement(document.body, {
                        delimiters: [
                            {left: '\\\$begin:math:text$', right: '\\\\\\$end:math:text$', display: false},
                            {left: '\\\$begin:math:display$', right: '\\\\\\$end:math:display$', display: true}
                        ],
                        throwOnError: false
                    });
                    """
                )

                await page.pdf(path=output_pdf_path, scale=scale, print_background=True)
                await browser.close()

                try:
                    reader = pypdf.PdfReader(output_pdf_path)
                    if len(reader.pages) == 1:
                        print(f"Successfully rendered as a single page PDF with scale factor {scale}")
                        return True
                    else:
                        print(f"PDF has {len(reader.pages)} pages with scale factor {scale}, trying a smaller scale...")
                except Exception as pdf_check_error:
                    print(f"Error checking PDF page count: {pdf_check_error}")
                    return False

        except Exception as e:
            print(f"Error rendering PDF with Playwright at scale {scale}: {e}")
    print("Failed to render PDF as a single page with any scale factor")
    return False


def process_pdf(pdf_info, args, client):
    """
    Process a single PDF from a local folder:
      - Select a random page from the PDF.
      - Render that page as a base64 PNG.
      - Generate HTML from the image using Claude.
      - Optionally render the HTML to a PDF using Playwright.
      - Generate order tests from the HTML.
    """
    local_pdf_path, index = pdf_info
    original_pdf_name = os.path.basename(local_pdf_path)
    pdf_id = original_pdf_name
    temp_pdf_dir = os.path.join(args.temp_dir, f"{os.path.splitext(original_pdf_name)[0]}")
    os.makedirs(temp_pdf_dir, exist_ok=True)

    try:
        reader = pypdf.PdfReader(local_pdf_path)
        num_pages = len(reader.pages)
        if num_pages == 0:
            print(f"PDF has no pages: {local_pdf_path}")
            return None
        page_num = random.randint(1, num_pages)
    except Exception as e:
        print(f"Error reading {local_pdf_path}: {e}")
        return None

    try:
        image_base64 = render_pdf_to_base64png(local_pdf_path, page_num, target_longest_image_dim=2048)
    except Exception as e:
        print(f"Error rendering page {page_num} from {local_pdf_path}: {e}")
        return None

    try:
        html_content = generate_html_from_image(client, image_base64)
        if not html_content:
            print(f"Failed to generate HTML for {local_pdf_path}, page {page_num}")
            return None
    except Exception as e:
        print(f"Error generating HTML for {local_pdf_path}: {e}")
        return None

    html_dir = os.path.join(args.output_dir, "html")
    pdfs_dir = os.path.join(args.output_dir, "pdfs")
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(pdfs_dir, exist_ok=True)

    html_path = os.path.join(html_dir, f"{os.path.splitext(original_pdf_name)[0]}_page{page_num}.html")
    try:
        with open(html_path, "w") as f:
            f.write(html_content)
    except Exception as e:
        print(f"Error saving HTML for {local_pdf_path}: {e}")
        return None

    playwright_pdf_path = None
    render_success = False
    playwright_pdf_filename = f"{os.path.splitext(original_pdf_name)[0]}_page{page_num}.pdf"
    if not args.skip_playwright:
        playwright_pdf_path = os.path.join(pdfs_dir, playwright_pdf_filename)
        try:
            png_width, png_height = get_png_dimensions_from_base64(image_base64)
            render_success = asyncio.run(render_pdf_with_playwright(html_content, playwright_pdf_path, png_width, png_height))
            if render_success:
                print(f"Successfully rendered with Playwright: {playwright_pdf_path}")
            else:
                print(f"Failed to render as a single page PDF: {playwright_pdf_path}")
                playwright_pdf_path = None
        except Exception as e:
            print(f"Failed to render with Playwright: {e}")
            playwright_pdf_path = None
            render_success = False
    if not args.skip_playwright and not render_success:
        return None

    tests = generate_tests_from_html(html_content, pdf_id, page_num)
    # IMPORTANT: Preserve the original PDF filename in the tests.
    for test in tests:
        test["pdf"] = original_pdf_name

    try:
        if os.path.exists(temp_pdf_dir):
            subprocess.run(["rm", "-rf", temp_pdf_dir])
    except Exception as e:
        print(f"Error cleaning up temp directory {temp_pdf_dir}: {e}")

    return {
        "pdf_id": pdf_id,
        "pdf_path": local_pdf_path,
        "page_number": page_num,
        "html_path": html_path,
        "playwright_pdf_path": playwright_pdf_path,
        "tests": tests,
        "num_tests": len(tests),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs in a folder to HTML templates and render with Playwright (order tests only)")
    parser.add_argument("--input_dir", required=True, help="Folder containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Directory to store HTML and tests")
    parser.add_argument("--temp_dir", default="/tmp/mine_tables", help="Directory for temporary files")
    parser.add_argument("--max_tests", type=int, default=100, help="Maximum number of PDFs to process (randomly selected)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel threads to use")
    parser.add_argument("--api_key", help="Claude API key (or set ANTHROPIC_API_KEY environment variable)")
    parser.add_argument("--skip_playwright", action="store_true", help="Skip Playwright PDF rendering")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: API key not provided. Use --api_key or set ANTHROPIC_API_KEY environment variable.")
        return

    client = Anthropic(api_key=api_key)

    pdf_paths = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))
    print(f"Found {len(pdf_paths)} PDF files in {args.input_dir}")

    random.shuffle(pdf_paths)
    pdf_paths = pdf_paths[: args.max_tests]

    synthetic_json_path = os.path.join(args.output_dir, "synthetic.jsonl")
    open(synthetic_json_path, "w").close()

    test_counter = 0
    test_types = {"order": 0}
    results = []

    import threading

    file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process_pdf, (pdf_path, i), args, client): pdf_path for i, pdf_path in enumerate(pdf_paths)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing PDFs"):
            pdf_path = futures[future]
            try:
                result = future.result()
                if result and result.get("tests"):
                    results.append(result)
                    with file_lock:
                        with open(synthetic_json_path, "a") as f:
                            for test in result["tests"]:
                                f.write(json.dumps(test) + "\n")
                        test_counter += len(result["tests"])
                        for test in result["tests"]:
                            if test.get("type") == "order":
                                test_types["order"] += 1
                        print(f"Added {len(result['tests'])} tests from {result['pdf_id']}, total: {test_counter}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    print(f"Generated {len(results)} HTML templates")
    if not args.skip_playwright:
        playwright_success = sum(1 for r in results if r and r.get("playwright_pdf_path"))
        print(f"Playwright PDF rendering: {playwright_success}/{len(results)} successful")
    print(f"Saved {test_counter} tests to {synthetic_json_path}")
    print(f"Generated a total of {test_counter} tests across {len(results)} templates")
    print("Test type distribution:")
    for test_type, count in test_types.items():
        print(f"  - {test_type}: {count} tests")


if __name__ == "__main__":
    main()
