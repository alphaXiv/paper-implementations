#!/usr/bin/env python3
"""
mine_math.py - Extract and validate math equations from candidate files and TeX bases.

This upgraded version:
  • Uses the Python logging module for cleaner logging.
  • Uses tqdm to display a progress bar.
  • Uses ProcessPoolExecutor to process TeX file groups in parallel.
  • For each TeX file, shuffles its pages randomly and processes them one-by-one.
    Once three pages return at least one equation each, further pages are skipped.
  • Adds an argparse argument for the similarity threshold for matches.
  • Saves JSONL outputs incrementally as each TeX file group is processed.

Usage:
  python mine_math.py --math_data /path/to/math_data --candidate candidate_folder --output_file math_tests.jsonl
    [--max_pages 3] [--parallel 8] [--sim_threshold 0.7]
"""

import argparse
import glob
import logging
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numba
import numpy as np
from tqdm import tqdm

from olmocr.bench.katex.render import render_equation
from olmocr.bench.tests import (
    MathTest,  # Assumes MathTest is JSON serializable or has __dict__
)
from olmocr.bench.tests import (
    save_tests,  # Original saving function (not used for incremental save)
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --- Utility Functions ---


def normalize_text(text: str) -> str:
    """Normalize text for better matching."""
    text = re.sub(r"\s+", " ", text)
    replacements = {"'": "'", "‚": "'", '"': '"', "„": '"', "＿": "_", "–": "-", "—": "-", "‑": "-", "‒": "-"}
    for fancy_char, ascii_char in replacements.items():
        text = text.replace(fancy_char, ascii_char)
    return text


def extract_tex_content(tex_file: str) -> str:
    """Extract the content from a TeX file."""
    try:
        with open(tex_file, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(tex_file, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            logging.error("Error reading %s: %s", tex_file, e)
            return ""


def extract_candidate_content(candidate_file: str) -> str:
    """Extract the content from a candidate .md file."""
    try:
        with open(candidate_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error("Error reading %s: %s", candidate_file, e)
        return ""


def extract_math_from_tex(tex_content: str) -> List[Tuple[str, str]]:
    """
    Extract math equations from TeX content.
    Returns list of tuples (equation_type, equation_content)
    """
    math_equations = []

    # Patterns for display math
    display_patterns = [
        (r"\$\$(.*?)\$\$", "$$"),
        (r"\\begin\{equation\}(.*?)\\end\{equation\}", "equation"),
        (r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}", "equation*"),
        (r"\\begin\{align\}(.*?)\\end\{align\}", "align"),
        (r"\\begin\{align\*\}(.*?)\\end\{align\*\}", "align*"),
        (r"\\begin\{displaymath\}(.*?)\\end\{displaymath\}", "displaymath"),
        (r"\\\[(.*?)\\\]", "displaymath"),
    ]
    # Patterns for inline math
    inline_patterns = [(r"\$(.*?)\$", "inline"), (r"\\\((.*?)\\\)", "inline")]

    for pattern_list in [display_patterns, inline_patterns]:
        for pattern, eq_type in pattern_list:
            matches = re.finditer(pattern, tex_content, re.DOTALL)
            for match in matches:
                equation = match.group(1).strip()
                if equation and not equation.isspace():
                    math_equations.append((eq_type, equation))
    return math_equations


@numba.njit
def compute_dp(candidate_arr, text_arr):
    m = candidate_arr.shape[0]
    n = text_arr.shape[0]
    dp = np.empty((m + 1, n + 1), dtype=np.int32)
    # For empty candidate, cost is 0 (can match anywhere in text)
    for j in range(n + 1):
        dp[0, j] = 0
    # When text is empty, need to delete all candidate characters.
    for i in range(1, m + 1):
        dp[i, 0] = i

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if candidate_arr[i - 1] == text_arr[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j - 1] + cost, dp[i - 1, j] + 1, dp[i, j - 1] + 1  # substitution or match  # deletion (from candidate)
            )  # insertion (in candidate)
    return dp


@numba.njit
def find_best_end(dp, m, n):
    best_distance = 1 << 30  # a large number
    best_end = 0
    for j in range(n + 1):
        if dp[m, j] < best_distance:
            best_distance = dp[m, j]
            best_end = j
    return best_end, best_distance


@numba.njit
def backtrack(dp, candidate_arr, text_arr, m, best_end):
    i = m
    j = best_end
    while i > 0:
        # Check for a diagonal move (match or substitution)
        if j > 0 and dp[i, j] == dp[i - 1, j - 1] + (0 if candidate_arr[i - 1] == text_arr[j - 1] else 1):
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return j  # start index in text


def find_matching_content(candidate_text: str, tex_content: str, sim_threshold: float) -> Optional[str]:
    """
    Find the substring of tex_content that most closely matches candidate_text using
    dynamic programming accelerated by numba. Returns the matching substring if its
    normalized similarity (1 - (edit_distance / len(candidate_text))) is above sim_threshold,
    otherwise returns None.
    """
    candidate_norm = normalize_text(candidate_text)
    tex_norm = normalize_text(tex_content)

    m = len(candidate_norm)
    n = len(tex_norm)
    if m == 0 or n == 0:
        return None

    # Convert strings to numpy arrays of integer character codes.
    candidate_arr = np.empty(m, dtype=np.int32)
    for i, c in enumerate(candidate_norm):
        candidate_arr[i] = ord(c)
    text_arr = np.empty(n, dtype=np.int32)
    for j, c in enumerate(tex_norm):
        text_arr[j] = ord(c)

    dp = compute_dp(candidate_arr, text_arr)
    best_end, min_distance = find_best_end(dp, m, n)
    similarity = (m - min_distance) / m

    logging.info("Similarity: %.3f", similarity)
    if similarity < sim_threshold:
        return None
    start_index = backtrack(dp, candidate_arr, text_arr, m, best_end)
    return tex_norm[start_index:best_end]


def parse_candidate_filename(filename: str) -> Optional[Tuple[str, int]]:
    """
    Parse candidate filename in the format: [tex file basename]_pg[pagenum]_repeat1.md
    Returns tuple (tex_basename, page_num) or None if the format doesn't match.
    """
    basename = os.path.basename(filename)
    match = re.match(r"(.+)_pg(\d+)_repeat\d+\.md$", basename)
    if match:
        tex_basename = match.group(1)
        page_num = int(match.group(2))
        return tex_basename, page_num
    return None


def validate_equation(equation: str) -> bool:
    """
    Validate that an equation renders correctly with KaTeX.
    Returns True if the equation is valid, False otherwise.
    """
    rendered = render_equation(equation)
    return rendered is not None


def process_candidate_file(candidate_file: str, pdfs_folder: str, sim_threshold: float) -> List[MathTest]:
    """
    Process a single candidate file.
    Returns a list of MathTest objects extracted from the corresponding TeX file.
    """
    logging.info("Processing %s", candidate_file)
    tests = []
    parse_result = parse_candidate_filename(candidate_file)
    if not parse_result:
        logging.error("Filename %s does not match expected format.", candidate_file)
        return tests

    tex_basename, page_num = parse_result
    tex_file_path = os.path.join(pdfs_folder, f"{tex_basename}.tex")

    if not os.path.exists(tex_file_path):
        logging.error("TeX file %s not found for candidate %s.", tex_file_path, candidate_file)
        return tests

    candidate_text = extract_candidate_content(candidate_file)
    tex_content = extract_tex_content(tex_file_path)
    if not tex_content or not candidate_text or len(tex_content.strip()) < 100 or len(candidate_text.strip()) < 100:
        logging.error("No content extracted from %s", tex_file_path)
        return tests

    matching_tex = find_matching_content(candidate_text, tex_content, sim_threshold)
    if not matching_tex:
        logging.warning("No matching TeX content found in %s for candidate %s", tex_file_path, candidate_file)
        return tests

    logging.debug("Matching TeX content: %s", matching_tex)

    math_equations = extract_math_from_tex(matching_tex)
    if not math_equations:
        logging.warning("No math equations found in matching content for candidate %s", candidate_file)
        return tests

    # Filter out equations that are too short, remove duplicates, and shuffle
    math_equations = [(eq_type, eq.strip()) for (eq_type, eq) in math_equations if len(eq.strip()) > 20]
    math_equations = list(set(math_equations))
    random.shuffle(math_equations)

    for i, (eq_type, equation) in enumerate(math_equations):
        if validate_equation(equation):
            test_id = f"{tex_basename}_pg{page_num}_math_{i:03d}"
            math_test = MathTest(
                id=test_id,
                pdf=f"{tex_basename}.pdf",
                page=page_num,
                type="math",
                math=equation,
            )
            tests.append(math_test)
            if len(tests) >= 10:
                break

    return tests


def process_tex_file_group(tex_basename: str, candidate_files: List[str], pdfs_folder: str, sim_threshold: float, max_pages: int) -> List[MathTest]:
    """
    For a given TeX file, group candidate files by page, randomly shuffle the pages,
    and process them one-by-one. Stop once max_pages (pages with valid equations) have
    been processed.
    """
    tests = []
    valid_pages = set()

    # Group candidate files by page number.
    page_dict: Dict[int, List[str]] = {}
    for candidate_file in candidate_files:
        parse_result = parse_candidate_filename(candidate_file)
        if not parse_result:
            continue
        _, page_num = parse_result
        page_dict.setdefault(page_num, []).append(candidate_file)

    # For each page, randomly choose one candidate file.
    distinct_candidate_files = []
    for page_num, files in page_dict.items():
        chosen_file = random.choice(files)
        distinct_candidate_files.append(chosen_file)

    # Shuffle the pages randomly.
    random.shuffle(distinct_candidate_files)

    # Process pages sequentially until max_pages with valid equations have been found.
    for candidate_file in distinct_candidate_files:
        result = process_candidate_file(candidate_file, pdfs_folder, sim_threshold)
        if result:
            tests.extend(result)
            # Mark this page as valid.
            page_num = parse_candidate_filename(candidate_file)[1]
            valid_pages.add(page_num)
            if len(valid_pages) >= max_pages:
                break

    return tests


def main():
    parser = argparse.ArgumentParser(description="Extract math equations from candidate files and corresponding TeX bases.")
    parser.add_argument("--math_data", required=True, help="Path to math_data folder")
    parser.add_argument("--candidate", required=True, help="Candidate folder name inside math_data")
    parser.add_argument("--max_pages", type=int, default=1, help="Maximum distinct pages with equations to process per TeX document")
    parser.add_argument("--parallel", type=int, default=8, help="Maximum process pool workers")
    parser.add_argument("--sim_threshold", type=float, default=0.7, help="Similarity threshold for matching candidate text")

    args = parser.parse_args()

    candidate_folder = os.path.join(args.math_data, args.candidate)
    pdfs_folder = os.path.join(args.math_data, "pdfs")

    candidate_files = glob.glob(os.path.join(candidate_folder, "*.md"))
    logging.info("Found %d candidate files.", len(candidate_files))

    # Group candidate files by TeX basename.
    tex_groups: Dict[str, List[str]] = {}
    for candidate_file in candidate_files:
        parse_result = parse_candidate_filename(candidate_file)
        if not parse_result:
            continue
        tex_basename, _ = parse_result
        tex_groups.setdefault(tex_basename, []).append(candidate_file)
    logging.info("Found %d TeX groups.", len(tex_groups))

    # Remove output file if it exists to start fresh
    output_file = os.path.join(args.math_data, "math_tests.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)

    all_math_tests = []

    # Process each TeX group in parallel using ProcessPoolExecutor.
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        future_to_tex = {
            executor.submit(process_tex_file_group, tex_basename, candidate_list, pdfs_folder, args.sim_threshold, args.max_pages): tex_basename
            for tex_basename, candidate_list in tex_groups.items()
        }
        for future in tqdm(as_completed(future_to_tex), total=len(future_to_tex), desc="Processing TeX files"):
            tex_basename = future_to_tex[future]
            try:
                tests = future.result()
                all_math_tests.extend(tests)
                # Incrementally save tests as each TeX group finishes processing.
                save_tests(all_math_tests, output_file)
            except Exception as e:
                logging.error("Error processing TeX group %s: %s", tex_basename, e)

    logging.info("Found %d valid math equations from %d TeX groups.", len(all_math_tests), len(tex_groups))
    logging.info("Results incrementally saved to %s", output_file)


if __name__ == "__main__":
    main()
