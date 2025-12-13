#!/usr/bin/env python3
"""
Batch OCR Processing Script
Handles multiple PDFs through the separated tokenization and generation pipeline
"""

import os
import argparse
import subprocess
import glob
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*50)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        print(f"Error: {e.stderr}")
        return False


def tokenize_all_pdfs(pdf_dir, tokenized_dir, batch_size=20, max_workers=4):
    """Tokenize all PDFs in a directory using threading"""
    pdf_files = glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return False

    print(f"Found {len(pdf_files)} PDF files")

    # Create batches
    batches = [pdf_files[i:i + batch_size] for i in range(0, len(pdf_files), batch_size)]

    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)} ({len(batch)} PDFs)")

        # Create batch output directory
        batch_output_dir = os.path.join(tokenized_dir, f"batch_{i+1:03d}")
        os.makedirs(batch_output_dir, exist_ok=True)

        # Use threading to parallelize tokenization
        def tokenize_single_pdf(pdf_path):
            pdf_name = Path(pdf_path).stem
            cmd = [
                "python", "tokenize_ocr_data.py",
                "--input", pdf_path,
                "--output", batch_output_dir,
                "--single"
            ]

            success = run_command(cmd, f"Tokenize {pdf_name}")
            if not success:
                print(f"Failed to tokenize {pdf_path}")
            return success

        print(f"Starting tokenization with {max_workers} threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tokenization tasks
            future_to_pdf = {executor.submit(tokenize_single_pdf, pdf_path): pdf_path for pdf_path in batch}

            # Wait for completion and track progress
            completed = 0
            for future in as_completed(future_to_pdf):
                completed += 1
                pdf_path = future_to_pdf[future]
                try:
                    success = future.result()
                    if success:
                        print(f"✓ Completed {completed}/{len(batch)}: {Path(pdf_path).name}")
                    else:
                        print(f"✗ Failed {completed}/{len(batch)}: {Path(pdf_path).name}")
                except Exception as e:
                    print(f"✗ Exception in {Path(pdf_path).name}: {e}")

        print(f"Batch {i+1} tokenization completed")

    return True


def generate_all_outputs(tokenized_dir, output_dir, batch_size=50):
    """Generate OCR outputs for all tokenized data"""
    batch_dirs = glob.glob(os.path.join(tokenized_dir, "batch_*"))

    if not batch_dirs:
        print(f"No tokenized batches found in {tokenized_dir}")
        return False

    print(f"Found {len(batch_dirs)} tokenized batches")

    for batch_dir in sorted(batch_dirs):
        batch_name = os.path.basename(batch_dir)
        print(f"\nProcessing {batch_name}")

        tokenized_files = glob.glob(os.path.join(batch_dir, "*_tokenized.pkl"))

        for tokenized_file in tokenized_files:
            pdf_name = Path(tokenized_file).stem.replace('_tokenized', '')
            output_subdir = os.path.join(output_dir, batch_name, pdf_name)

            cmd = [
                "python", "generate_ocr_outputs.py",
                "--tokenized-file", tokenized_file,
                "--output", output_subdir
            ]

            if not run_command(cmd, f"Generate OCR for {pdf_name}"):
                print(f"Failed to generate output for {tokenized_file}, continuing...")

    return True


def main():
    parser = argparse.ArgumentParser(description="Batch OCR processing pipeline")
    parser.add_argument("--pdf-dir", "-p", required=True, help="Directory containing PDF files")
    parser.add_argument("--tokenized-dir", "-t", default="./tokenized_data", help="Directory for tokenized data")
    parser.add_argument("--output-dir", "-o", default="./ocr_outputs", help="Directory for OCR outputs")
    parser.add_argument("--tokenize-batch-size", "-tb", type=int, default=20, help="Batch size for tokenization")
    parser.add_argument("--generate-batch-size", "-gb", type=int, default=50, help="Batch size for generation")
    parser.add_argument("--max-workers", "-w", type=int, default=4, help="Maximum number of parallel tokenization threads")
    parser.add_argument("--phase", choices=["tokenize", "generate", "both"], default="both",
                       help="Which phase to run")

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.tokenized_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    success = True

    if args.phase in ["tokenize", "both"]:
        print("\n🚀 Starting Tokenization Phase")
        if not tokenize_all_pdfs(args.pdf_dir, args.tokenized_dir, args.tokenize_batch_size, args.max_workers):
            success = False

    if args.phase in ["generate", "both"]:
        print("\n🤖 Starting Generation Phase")
        if not generate_all_outputs(args.tokenized_dir, args.output_dir, args.generate_batch_size):
            success = False

    if success:
        print("\n✅ Batch processing completed successfully!")
        print(f"Tokenized data: {args.tokenized_dir}")
        print(f"OCR outputs: {args.output_dir}")
    else:
        print("\n❌ Batch processing completed with errors")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())