#!/usr/bin/env python3
"""Convert OLMOCR JSONL output to markdown files"""

import json
import os
from pathlib import Path

# Get the OmniDocBench-Evals root directory (3 levels up from this script)
root_dir = Path(__file__).parent.parent.parent.parent

# Paths
results_dir = root_dir / "outputs/olmocr_workspace/results"
markdown_dir = root_dir / "outputs/olmocr_workspace/markdown"
markdown_dir.mkdir(exist_ok=True)

# Process all JSONL files
total_converted = 0
for jsonl_file in results_dir.glob("*.jsonl"):
    print(f"Processing {jsonl_file.name}...")
    with open(jsonl_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            doc = json.loads(line)
            
            # Extract the PDF filename from metadata
            if 'metadata' in doc and 'Source-File' in doc['metadata']:
                pdf_path = doc['metadata']['Source-File']
                # Get just the filename without extension
                pdf_name = Path(pdf_path).stem
                
                # Create markdown filename
                md_filename = f"{pdf_name}.md"
                md_path = markdown_dir / md_filename
                
                # Write the text content to markdown
                with open(md_path, 'w') as md_file:
                    md_file.write(doc.get('text', ''))
                
                total_converted += 1
            else:
                print(f"Warning: No Source-File metadata in doc: {doc.get('id', 'unknown')}")

print(f"\nConversion complete! Created {total_converted} markdown files in {markdown_dir}")
