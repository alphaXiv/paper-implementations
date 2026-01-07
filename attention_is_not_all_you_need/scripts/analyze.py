"""
Simple analysis script for training results.
"""

import argparse
import os
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("Analyzing results...")

    # Look for results.json files
    for json_file in results_dir.rglob("results.json"):
        print(f"Found results: {json_file}")
        with open(json_file, 'r') as f:
            results = json.load(f)
            print(json.dumps(results, indent=2))

    print("Analysis complete.")

if __name__ == "__main__":
    main()