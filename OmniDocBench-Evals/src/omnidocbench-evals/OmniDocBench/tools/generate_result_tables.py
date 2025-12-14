#!/usr/bin/env python3
"""
Generate OCR evaluation result tables from JSON metric files.
This script processes OmniDocBench evaluation results and generates formatted tables.
"""

import os
import pandas as pd
import numpy as np
import json
import argparse
import sys

def load_ocr_results(ocr_type, result_folder, match_name):
    """Load OCR results from JSON file and extract metrics."""
    result_path = os.path.join(result_folder, f'{ocr_type}_{match_name}_metric_result.json')
    print(f"Loading results from: {result_path}")

    if not os.path.exists(result_path):
        print(f"Warning: Result file not found: {result_path}")
        return None

    with open(result_path, 'r') as f:
        result = json.load(f)

    save_dict = {}

    # Define the metrics to extract
    metrics = [
        ("text_block", "Edit_dist"),
        ("display_formula", "Edit_dist"),
        ("table", "TEDS"),
        ("table", "TEDS_structure_only"),
        ("reading_order", "Edit_dist")
    ]

    for category_type, metric in metrics:
        if metric in ['CDM', 'TEDS', 'TEDS_structure_only']:
            if result[category_type]["page"].get(metric):
                save_dict[f'{category_type}_{metric}'] = result[category_type]["page"][metric]["ALL"] * 100
            else:
                save_dict[f'{category_type}_{metric}'] = 0
        else:
            save_dict[f'{category_type}_{metric}'] = result[category_type]["all"][metric].get("ALL_page_avg", np.nan)

    return save_dict

def calculate_overall_score(df):
    """Calculate overall score from individual metrics."""
    # Overall = ((1-text_block_Edit_dist)*100 + (1-display_formula_Edit_dist)*100 + table_TEDS)/3
    df_copy = df.copy()
    df_copy['overall'] = (
        (1 - df_copy['text_block_Edit_dist']) * 100 +
        (1 - df_copy['display_formula_Edit_dist']) * 100 +
        df_copy['table_TEDS']
    ) / 3
    return df_copy

def generate_table(ocr_types_dict, result_folder, match_name, output_format='table'):
    """Generate results table for specified OCR types."""
    dict_list = []

    for ocr_name, ocr_type in ocr_types_dict.items():
        results = load_ocr_results(ocr_type, result_folder, match_name)
        if results:
            dict_list.append(results)
        else:
            print(f"Skipping {ocr_name} due to missing results")

    if not dict_list:
        print("No valid results found!")
        return

    df = pd.DataFrame(dict_list, index=ocr_types_dict.keys()).round(3)
    df = calculate_overall_score(df)

    if output_format == 'csv':
        print(df.to_csv())
    elif output_format == 'json':
        print(df.to_json(orient='index', indent=2))
    else:
        # Pretty print the table
        print("\nOCR Evaluation Results:")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Generate OCR evaluation result tables')
    parser.add_argument('--result-folder', default='../result',
                       help='Path to the result folder containing JSON files')
    parser.add_argument('--match-name', default='quick_match',
                       choices=['no_split', 'quick_match'],
                       help='Match name for result files')
    parser.add_argument('--ocr-type', default='chandra_ocr_chinese',
                       choices=[
                           'results_dpsk_uncleaned/results_dpsk',
                           'dpsk_cleaned_results/results_dpsk-cleaned',
                           'olmo_ocr_2_results/images_to_pdf',
                           'olmo_ocr_2_results_english/images_to_pdf',
                           'dpsk_results_english/results_dpsk-cleaned',
                           'dpsk_results_chinese/results_dpsk-cleaned',
                           'chandra_ocr_chinese/output_results_markdown',
                           'chandra_ocr_english/output_results_markdown',
                           'chandra_ocr/output_results_markdown',
                           'olmo_ocr_2_results_chinese/images_to_pdf'
                       ],
                       help='OCR type to process')
    parser.add_argument('--output-format', default='table',
                       choices=['table', 'csv', 'json'],
                       help='Output format')
    parser.add_argument('--all-ocr-types', action='store_true',
                       help='Process all OCR types')

    args = parser.parse_args()

    # Define OCR types dictionary
    if args.all_ocr_types:
        ocr_types_dict = {
            'DeepSeek-OCR-Uncleaned': 'results_dpsk_uncleaned/results_dpsk',
            'DeepSeek-OCR-Cleaned': 'dpsk_cleaned_results/results_dpsk-cleaned',
            'OLMO-OCR-2': 'olmo_ocr_2_results/images_to_pdf',
            'OLMO-OCR-2-English': 'olmo_ocr_2_results_english/images_to_pdf',
            'DeepSeek-OCR-English': 'dpsk_results_english/results_dpsk-cleaned',
            'DeepSeek-OCR-Chinese': 'dpsk_results_chinese/results_dpsk-cleaned',
            'Chandra-OCR-Chinese': 'chandra_ocr_chinese/output_results_markdown',
            'Chandra-OCR-English': 'chandra_ocr_english/output_results_markdown',
            'Chandra-OCR': 'chandra_ocr/output_results_markdown',
            'OLMO-OCR-2-Chinese': 'olmo_ocr_2_results_chinese/images_to_pdf'
        }
    else:
        ocr_types_dict = {
            args.ocr_type.split('/')[-1].replace('_', '-').title(): args.ocr_type
        }

    # Check if result folder exists
    if not os.path.exists(args.result_folder):
        print(f"Error: Result folder not found: {args.result_folder}")
        sys.exit(1)

    generate_table(ocr_types_dict, args.result_folder, args.match_name, args.output_format)

if __name__ == '__main__':
    main()</content>
<parameter name="filePath">/home/ubuntu/texas-sandbox/paper-implementations/OmniDocBench-Evals/src/omnidocbench-evals/OmniDocBench/tools/generate_result_tables.py