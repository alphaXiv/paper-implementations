import json
import random
import re


def extract_random_segment(text, min_words=7, max_words=15):
    """Extract a random segment of 7-15 words from the text."""
    words = text.split()
    if len(words) <= max_words:
        return text  # Return full text if it's shorter than max_words

    max_start = len(words) - min_words
    start = random.randint(0, max_start)
    remaining_words = len(words) - start
    segment_length = random.randint(min_words, min(max_words, remaining_words))
    segment = words[start : start + segment_length]
    return " ".join(segment)


def process_jsonl_file_present(input_file, output_file):
    """Process a JSONL file and create multiple random cases for each PDF."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                image = data["image"]
                original_text = data["text"]
                num_cases = random.randint(1, 3)

                for _ in range(num_cases):
                    processed_num = random.randint(5, 10)
                    processed_id = f"{image}_processed{processed_num:02d}"
                    max_diffs = random.randint(1, 2)
                    text_segment = extract_random_segment(original_text)

                    new_case = {
                        "pdf": f"{image}.pdf",
                        "page": 1,
                        "id": processed_id,
                        "type": "present",
                        "max_diffs": max_diffs,
                        "text": text_segment,
                        "case_sensitive": True,
                        "first_n": None,
                        "last_n": None,
                    }
                    outfile.write(json.dumps(new_case) + "\n")


def extract_ordered_segments(text, min_words=7, max_words=15):
    """Extract two ordered segments from the text."""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) < 2:
        return None, None
    valid_indices = list(range(len(sentences)))
    if len(valid_indices) <= 2:
        before_idx, after_idx = 0, 1
    else:
        before_idx = random.randint(0, len(valid_indices) - 2)
        after_idx = random.randint(before_idx + 1, len(valid_indices) - 1)
    before_sentence = sentences[before_idx]
    after_sentence = sentences[after_idx]

    before_words = before_sentence.split()
    after_words = after_sentence.split()

    if len(before_words) > max_words:
        start = random.randint(0, len(before_words) - min_words)
        length = random.randint(min_words, min(max_words, len(before_words) - start))
        before_segment = " ".join(before_words[start : start + length])
    else:
        before_segment = before_sentence

    if len(after_words) > max_words:
        start = random.randint(0, len(after_words) - min_words)
        length = random.randint(min_words, min(max_words, len(after_words) - start))
        after_segment = " ".join(after_words[start : start + length])
    else:
        after_segment = after_sentence

    return before_segment, after_segment


def process_jsonl_file_order(input_file, output_file):
    """Process a JSONL file and create order-type cases."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                image = data["image"]
                original_text = data["text"]
                num_cases = random.randint(1, 3)

                for _ in range(num_cases):
                    before_text, after_text = extract_ordered_segments(original_text)
                    if not before_text or not after_text:
                        continue
                    processed_num = random.randint(11, 16)
                    processed_id = f"{image}_processed{processed_num:02d}"
                    max_diffs = random.randint(1, 3)

                    new_case = {
                        "pdf": f"{image}.pdf",
                        "page": 1,
                        "id": processed_id,
                        "type": "order",
                        "before": before_text,
                        "after": after_text,
                        "max_diffs": max_diffs,
                        "checked": "verified",
                        "url": f"https://example.com/document/{image}",
                    }

                    outfile.write(json.dumps(new_case) + "\n")


if __name__ == "__main__":
    input_file = "olmoce/bench/sample_data/old_scans.jsonl"
    output_file = "order_cases.jsonl"
    process_jsonl_file_present(input_file, output_file)
    process_jsonl_file_order(input_file, output_file)
