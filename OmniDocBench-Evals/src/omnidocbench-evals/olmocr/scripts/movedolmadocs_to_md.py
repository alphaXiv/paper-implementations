#!/usr/bin/env python3
import argparse
import json
import os
from urllib.parse import urlparse

import boto3


def parse_args():
    parser = argparse.ArgumentParser(description="Read JSONL files from an S3 prefix, extract text, and write to local .md files.")
    parser.add_argument(
        "--s3-prefix",
        default="s3://ai2-oe-data/jakep/pdfworkspaces/pdelfin_testset/results/",
        help="S3 prefix containing the JSONL files (default: s3://ai2-oe-data/jakep/pdfworkspaces/pdelfin_testset/results/)",
    )
    parser.add_argument("--output-dir", default="output_md", help="Local directory to store output .md files (default: output_md)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse the s3-prefix into bucket and prefix
    parsed_s3 = urlparse(args.s3_prefix)
    # e.g. netloc = 'ai2-oe-data', path = '/jakep/pdfworkspaces/pdelfin_testset/results/'
    bucket_name = parsed_s3.netloc
    # Remove leading '/' from parsed_s3.path
    prefix = parsed_s3.path.lstrip("/")

    # Ensure local output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize S3 client
    s3 = boto3.client("s3")

    # List all objects under the prefix
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            # Skip non-jsonl files
            if not key.endswith(".jsonl"):
                continue

            print(f"Processing S3 object: s3://{bucket_name}/{key}")

            # Read the S3 object
            s3_object = s3.get_object(Bucket=bucket_name, Key=key)
            # s3_object['Body'] is a StreamingBody, so we can read it line-by-line
            body_stream = s3_object["Body"].iter_lines()

            for line in body_stream:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print("Warning: Failed to decode JSON line.")
                    continue

                # Extract text
                text_content = record.get("text", "")
                if not text_content.strip():
                    # If there's no text, skip
                    continue

                # Derive the output filename based on the "Source-File" metadata
                metadata = record.get("metadata", {})
                source_file = metadata.get("Source-File", "")

                # Example: source_file = 's3://ai2-oe-data/jakep/pdfdata/pdelfin_testset/fcffd2dd327d4e58d3c6d1d22ba62531c863_page8.pdf'
                # We want to end up with: 'fcffd2dd327d4e58d3c6d1d22ba62531c863_page8_pdelf.md'

                # 1) Extract just the filename from the path
                # 2) Remove '.pdf'
                # 3) Append '_pdelf.md'
                source_filename = os.path.basename(source_file)  # e.g. 'fcffd2dd327d4e58d3c6d1d22ba62531c863_page8.pdf'
                if source_filename.lower().endswith(".pdf"):
                    source_filename = source_filename[:-4]  # remove .pdf

                output_filename = f"{source_filename}_pdelf.md"
                output_path = os.path.join(args.output_dir, output_filename)

                # Append the text to the corresponding file
                # If you want to overwrite instead, change mode to 'w'
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(text_content + "\n")

                # Optional: Print or log what you've written
                # print(f"Appended text to {output_path}")

    print("Done processing all JSONL files.")


if __name__ == "__main__":
    main()
