"""
Convert JSONL files to Markdown files and handle missing PDFs
Usage:
    python workspace_to_benchmark.py localworkspace ./markdown_output --bench-path ../olmOCR-bench/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl_files(input_dir):
    """Load all JSONL files from the input directory."""
    jsonl_files = list(Path(input_dir).glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return []

    print(f"Found {len(jsonl_files)} JSONL files: {[f.name for f in jsonl_files]}")
    return jsonl_files


def parse_jsonl_entries(jsonl_files):
    """Parse all JSONL files and extract entries with text and metadata."""
    all_entries = []
    pdf_sources = set()

    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file.name}...")

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                entry = json.loads(line)
                text = entry.get("text", "")
                metadata = entry.get("metadata", {})
                source_file = metadata.get("Source-File", "")

                if source_file:
                    pdf_sources.add(source_file)

                all_entries.append({"text": text, "source_file": source_file, "metadata": metadata, "entry": entry})

    print(f"Loaded {len(all_entries)} entries from JSONL files")
    print(f"Found {len(pdf_sources)} unique PDF sources")

    return all_entries, pdf_sources


def get_subdir_and_pdf_name(source_file_path):
    """Extract subdirectory and PDF filename from source file path."""
    if not source_file_path:
        return None, None

    path_parts = Path(source_file_path).parts

    pdfs_index = path_parts.index("pdfs")
    if pdfs_index + 1 < len(path_parts):
        subdir = path_parts[pdfs_index + 1]
        pdf_name = Path(source_file_path).stem
        return subdir, pdf_name

    return None, None


def create_markdown_files(entries, output_dir, repeat_index=1):
    """Create markdown files from JSONL entries in subdir/{pdf_name}.md format."""
    output_path = Path(output_dir)

    subdir_pdf_to_entries = defaultdict(list)
    blank_files = 0

    for entry in entries:
        subdir, pdf_name = get_subdir_and_pdf_name(entry["source_file"])
        if subdir and pdf_name:
            key = (subdir, pdf_name)
            subdir_pdf_to_entries[key].append(entry)

    created_files = set()

    for (subdir, pdf_name), pdf_entries in subdir_pdf_to_entries.items():
        subdir_path = output_path / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

        md_filename = f"{pdf_name}_pg1_repeat{repeat_index}.md"
        md_filepath = subdir_path / md_filename

        assert len(pdf_entries) == 1, "Expecting just one entry mapping to each file, otherwise something is wrong"
        file_text = pdf_entries[0]["text"]

        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(file_text)

        if not file_text.strip():
            blank_files += 1

        created_files.add((subdir, pdf_name))
        print(f"Created: {subdir}/{md_filename}_pg1_repeat{repeat_index}")

    print(f"Created {len(created_files)} markdown files from JSONL data")
    print(f"{blank_files} of those had empty content")
    return created_files


def find_missing_pdfs(pdf_sources, created_files, base_bench_path):
    """Find PDFs that exist in directories but are missing from JSONL data."""
    subdirs = set()

    for source_file in pdf_sources:
        if not source_file:
            continue

        subdir, _ = get_subdir_and_pdf_name(source_file)
        if subdir:
            subdirs.add(subdir)

    print(f"Found PDF subdirectories: {sorted(subdirs)}")

    missing_pdfs = []

    for subdir in subdirs:
        pdf_dir = Path(base_bench_path) / "bench_data" / "pdfs" / subdir

        if not pdf_dir.exists():
            print(f"Warning: Directory {pdf_dir} does not exist")
            continue

        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in {subdir}/")

        for pdf_file in pdf_files:
            pdf_name = pdf_file.stem

            if (subdir, pdf_name) not in created_files:
                missing_pdfs.append({"pdf_name": pdf_name, "full_path": pdf_file, "subdir": subdir})

    print(f"Found {len(missing_pdfs)} missing PDFs")
    return missing_pdfs


def create_blank_markdown_files(missing_pdfs, output_dir, repeat_index=1):
    """Create blank markdown files for missing PDFs in subdir/{pdf_name}.md format."""
    output_path = Path(output_dir)

    for missing_pdf in missing_pdfs:
        subdir = missing_pdf["subdir"]
        pdf_name = missing_pdf["pdf_name"]

        subdir_path = output_path / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

        md_filename = f"{pdf_name}_pg1_repeat{repeat_index}.md"
        md_filepath = subdir_path / md_filename

        content = ""

        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Created blank: {subdir}/{md_filename}_pg1_repeat{repeat_index}")

    print(f"Created {len(missing_pdfs)} blank markdown files for missing PDFs")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL files to Markdown and handle missing PDFs")
    parser.add_argument("workspace_dir", help="Your workspace directory")
    parser.add_argument("output_dir", nargs="?", default="./markdown_output", help="Output directory for markdown files (default: ./markdown_output)")
    parser.add_argument("--bench-path", default="../olmOCR-bench", help="Path to olmOCR-bench directory (default: ../olmOCR-bench)")
    parser.add_argument(
        "--repeat-index", default=1, type=int, help="If you want to run multiple workspaces as different repeats to get a better average, set this"
    )

    args = parser.parse_args()
    input_dir = args.workspace_dir + "/results"
    input_dir = Path(input_dir)
    output_dir = Path(args.output_dir)
    bench_path = Path(args.bench_path)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    jsonl_files = load_jsonl_files(input_dir)
    if not jsonl_files:
        sys.exit(1)

    entries, pdf_sources = parse_jsonl_entries(jsonl_files)
    if not entries:
        print("No entries found in JSONL files")
        sys.exit(1)

    created_files = create_markdown_files(entries, output_dir)

    missing_pdfs = find_missing_pdfs(pdf_sources, created_files, bench_path)

    if missing_pdfs:
        create_blank_markdown_files(missing_pdfs, output_dir)

    print("\nSummary:")
    print(f"Created {len(created_files)} markdown files from JSONL data")
    print(f"Created {len(missing_pdfs)} blank markdown files for missing PDFs")
    print(f"Total markdown files: {len(created_files) + len(missing_pdfs)}")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
