#!/usr/bin/env python
import argparse
import glob
import json
import os

from datasets import load_dataset


def extract_urls_from_jsonl(file_path):
    """Extract URLs from a JSONL file."""
    urls = set()
    url_to_data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "url" in data and data["url"]:
                    url = data["url"]
                    urls.add(url)
                    # Store minimal context for each URL
                    url_to_data[url] = {"id": data.get("id", ""), "type": data.get("type", ""), "page": data.get("page", "")}
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON from line in {file_path}")
                continue
    return urls, url_to_data


def main():
    parser = argparse.ArgumentParser(description="Check for URL matches between local files and Hugging Face dataset")
    parser.add_argument("--local-dir", default="/home/ubuntu/olmocr/olmOCR-bench/bench_data", help="Directory containing local JSONL files")
    parser.add_argument("--output", default="url_matches.json", help="Output file for results")
    args = parser.parse_args()

    # Step 1: Get all local JSONL files
    local_jsonl_files = glob.glob(os.path.join(args.local_dir, "*.jsonl"))
    print(f"Found {len(local_jsonl_files)} local JSONL files.")

    # Step 2: Extract URLs from local files
    local_urls = {}
    all_local_urls = set()
    url_metadata = {}

    for file_path in local_jsonl_files:
        file_name = os.path.basename(file_path)
        urls, url_data = extract_urls_from_jsonl(file_path)
        local_urls[file_name] = urls
        all_local_urls.update(urls)

        # Store metadata with file information
        for url, data in url_data.items():
            if url not in url_metadata:
                url_metadata[url] = []
            url_metadata[url].append({"file": file_name, **data})

    print(f"Extracted {len(all_local_urls)} unique URLs from local files.")

    # Step 3: Load Hugging Face dataset
    print("Loading Hugging Face dataset...")
    try:
        dataset_documents = load_dataset("allenai/olmOCR-mix-0225", "00_documents")
        dataset_books = load_dataset("allenai/olmOCR-mix-0225", "01_books")

        # Step 4: Extract URLs from Hugging Face dataset
        hf_urls = set()

        for split in dataset_documents:
            for item in dataset_documents[split]:
                if "url" in item and item["url"]:
                    hf_urls.add(item["url"])

        for split in dataset_books:
            for item in dataset_books[split]:
                if "url" in item and item["url"]:
                    hf_urls.add(item["url"])

        print(f"Extracted {len(hf_urls)} unique URLs from Hugging Face dataset.")

        # Step 5: Find matches
        matches = all_local_urls.intersection(hf_urls)

        # Step 6: Group matches by local file with metadata
        matches_by_file = {}
        match_details = []

        for file_name, urls in local_urls.items():
            file_matches = urls.intersection(hf_urls)
            if file_matches:
                matches_by_file[file_name] = list(file_matches)

                # Add detailed metadata for each match
                for url in file_matches:
                    if url in url_metadata:
                        for entry in url_metadata[url]:
                            match_details.append({"url": url, "metadata": entry})

        # Print summary
        print(f"Found {len(matches)} matching URLs between local files and Hugging Face dataset.")

        for file_name, file_matches in matches_by_file.items():
            match_percentage = (len(file_matches) / len(local_urls[file_name])) * 100 if local_urls[file_name] else 0
            print(f"{file_name}: {len(file_matches)}/{len(local_urls[file_name])} matches ({match_percentage:.2f}%)")

        # Save results
        result = {
            "total_local_urls": len(all_local_urls),
            "total_hf_urls": len(hf_urls),
            "total_matches": len(matches),
            "matches_by_file": matches_by_file,
            "match_details": match_details,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to {args.output}")

    except Exception as e:
        print(f"Error loading or processing Hugging Face dataset: {e}")


if __name__ == "__main__":
    main()
