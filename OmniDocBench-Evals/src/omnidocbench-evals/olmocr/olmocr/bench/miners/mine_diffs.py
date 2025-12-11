import argparse
import base64
import os
import re
import time
from collections import Counter
from difflib import SequenceMatcher

import syntok.segmenter as segmenter
from google import genai
from google.genai import types

from olmocr.bench.tests import TextPresenceTest, save_tests
from olmocr.data.renderpdf import render_pdf_to_base64png

LABEL_WIDTH = 8  # fixed width for printing labels

# Uses a gemini prompt to get the most likely clean sentence from a pdf page
last_gemini_call = time.perf_counter()


def clean_base_sentence(pdf_path: str, page_num: int, base_sentence: str) -> str:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=2048)
    image_part = types.Part(inline_data=types.Blob(mime_type="image/png", data=base64.b64decode(image_base64)))
    model = "gemini-2.0-flash-thinking-exp-01-21"  # Consider using a more stable model for production
    # model="gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                image_part,
                types.Part.from_text(
                    text=f"""Base: {base_sentence}

Consider the sentence labeled "Base" above in the document image attached. What is the correct reading of this document within the image of the page? I need it to be exact down to the individual character and that's very important to get right. It needs to match the picture, not the provided text. Please just output the correct full sentence exactly how it appears in the document image and nothing else. You can merge hyphenated words back together, and don't output any new lines."""
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=500,
        response_mime_type="text/plain",
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    # Basic rate limitting
    global last_gemini_call
    if time.perf_counter() - last_gemini_call < 6:
        time.sleep(6 - (time.perf_counter() - last_gemini_call))

    last_gemini_call = time.perf_counter()

    # Return response
    if response is not None and response.candidates is not None and len(response.candidates) > 0:
        return response.candidates[0].content.parts[0].text
    else:
        return None


def parse_sentences(text: str) -> list[str]:
    """
    Splits a text into a list of sentence strings using syntok.
    Preserves original spacing and punctuation.
    """
    sentences = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            # Reconstruct the sentence with original spacing
            sentence_str = ""
            for token in sentence:
                sentence_str += token.spacing + token.value
            # Trim any leading whitespace
            sentence_str = sentence_str.lstrip()
            sentences.append(sentence_str)
    return sentences


def compare_votes_for_file(base_pdf_file: str, base_pdf_page: int, base_text: str, candidate_texts: list[str], max_diffs: int) -> None:
    """
    For each sentence in the base text, finds the best matching sentence from
    each candidate text (using a similarity threshold). If any candidate sentences
    differ from the base sentence, collects that diff (base sentence plus variant
    votes) for later printing. At the end, prints only the top N diffs (by total vote count)
    for the file.

    Comparison is case-insensitive, but output preserves original capitalization.
    """
    base_sentences = parse_sentences(base_text)
    # Parse all candidate texts into lists of sentences
    candidate_sentences_list = [parse_sentences(ct) for ct in candidate_texts]

    diffs = []  # list to hold diff entries
    for b_sentence in base_sentences:
        b_sentence = b_sentence.replace("\n", " ").strip()

        votes = []
        for c_sentences in candidate_sentences_list:
            best_ratio = 0.0
            best_candidate = None

            # Find the candidate sentence with the highest similarity to b_sentence
            # using case-insensitive comparison
            for c_sentence in c_sentences:
                ratio = SequenceMatcher(None, b_sentence.lower(), c_sentence.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_candidate = c_sentence  # Keep original capitalization for output

            # Append the candidate if it passes the similarity threshold (e.g., 0.7)
            if best_ratio > 0.5 and best_candidate is not None:
                votes.append(best_candidate.strip())

        # Only consider variants that differ when compared case-insensitively
        variant_votes = [vote for vote in votes if vote.lower() != b_sentence.lower()]
        if variant_votes:
            diff_entry = {
                "base": b_sentence,
                "variants": Counter(variant_votes),
                "vote_count": len(variant_votes),
            }
            diffs.append(diff_entry)

    # Sort diffs by vote_count descending and take only the top max_diffs
    diffs.sort(key=lambda d: d["vote_count"], reverse=True)
    top_diffs = diffs[:max_diffs]
    tests = []

    for index, diff in enumerate(top_diffs):
        base_sentence = diff["base"]
        variant_counter = diff["variants"]

        # Print base sentence using fixed-width label formatting
        print(f"{'Base:':<{LABEL_WIDTH}} {base_sentence}")
        print(f"{'Variants:':<{LABEL_WIDTH}}")
        for variant, count in variant_counter.items():
            label = f"{count}x:"
            print(f"{label:<{LABEL_WIDTH}} {variant}")
        # Get the clean version of the sentence
        cleaned = clean_base_sentence(base_pdf_file, base_pdf_page, base_sentence)
        print(f"{'Clean:':<{LABEL_WIDTH}} {cleaned}")
        print("-" * 40)

        if cleaned is None:
            cleaned = base_sentence

        tests.append(
            TextPresenceTest(
                pdf=os.path.basename(base_pdf_file),
                page=base_pdf_page,
                id=f"{os.path.basename(base_pdf_file).replace('.pdf', '')}_minediff_{index:02d}",
                type="present",
                threshold=1.0,
                text=cleaned,
            )
        )

    return tests


def get_pdf_from_md(md_path: str) -> str:
    base = os.path.basename(md_path)
    base = re.sub(r"_\d+\.md$", ".pdf", base)
    return os.path.join(os.path.dirname(md_path), "..", "pdfs", base)


def main():
    parser = argparse.ArgumentParser(description="Compares sentences from base and candidate texts, printing differences.")
    parser.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "chatgpt"), help="Path to the folder containing base .md files.")
    parser.add_argument("--compare", default=os.path.join(os.path.dirname(__file__), "olmocr"), help="Path to the folder containing candidate .md files.")
    parser.add_argument("--max-diffs", type=int, default=5, help="Maximum number of diffs to display per file.")
    parser.add_argument(
        "--output", default="mine_diffs_candidates.jsonl", type=str, help="Output of potential candidate test proposals, to be verified or added to dataset"
    )
    args = parser.parse_args()

    base_path = args.base
    compare_path = args.compare
    max_diffs = args.max_diffs

    # Collect all .md files from the base and compare folders
    base_files = [f for f in os.listdir(base_path) if f.endswith(".md")]

    all_tests = []

    # Process each base file and print out the vote differences
    for bf in base_files:
        base_file_path = os.path.join(base_path, bf)
        with open(base_file_path, "r", encoding="utf-8") as f:
            base_text = f.read()

        compare_files = [f for f in os.listdir(compare_path) if f.endswith(".md") and re.sub(r"_\d+\.md$", "", f) == re.sub(r"_\d+\.md$", "", bf)]

        if not compare_files:
            print(f"skipping {bf} nothing to compare against")

        # Read all candidate texts at once
        candidate_texts = []
        for cf in compare_files:
            with open(os.path.join(compare_path, cf), "r", encoding="utf-8") as f:
                candidate_texts.append(f.read())

        base_pdf_file = get_pdf_from_md(base_file_path)
        base_pdf_page = 1
        print(f"Results for base file: {bf}")
        tests = compare_votes_for_file(base_pdf_file, base_pdf_page, base_text, candidate_texts, max_diffs)
        all_tests.extend(tests)
        print("")

        # Output test candidates for review after each file, in case there are errors
        save_tests(all_tests, args.output)


if __name__ == "__main__":
    main()
