import os
import pandas as pd
import numpy as np
import json
import sys
import argparse


def process_args(args):
    parser = argparse.ArgumentParser(description='Process evaluation results and generate summary DataFrame.')
    parser.add_argument('--result-dir', type=str, default='./src/omnidocbench_evals/OmniDocBench/result',
                        help='Path to result directory containing metric_result.json files')
    parser.add_argument('--ocr-types', type=str, nargs='+', default=['deepseek', 'olmocr2'],
                        help='List of OCR types to process (default: deepseek olmocr2)')
    parser.add_argument('--language', type=str, default='all',
                        help='Language filter (all, english, simplified_chinese)')
    parser.add_argument('--match-method', type=str, default='quick_match',
                        help='Match method (default: quick_match)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (optional)')
    parameters = parser.parse_args(args)
    return parameters


def load_and_process_results(result_dir, ocr_types, language, match_method):
    """Load metric results and process them into a DataFrame."""
    dict_list = []
    ocr_names = []
    
    for ocr_type in ocr_types:
        # Construct file name based on naming convention: ocr_type_language_match_method_metric_result.json
        result_filename = f'{ocr_type}_{language}_{match_method}_metric_result.json'
        result_path = os.path.join(result_dir, result_filename)
        
        print(f"Loading: {result_path}")
        
        if not os.path.exists(result_path):
            print(f"Warning: Result file not found: {result_path}")
            continue
            
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
        except Exception as e:
            print(f"Error loading {result_path}: {e}")
            continue
        
        save_dict = {}
        
        # Extract metrics for each category
        for category_type, metric in [("text_block", "Edit_dist"), 
                                     ("display_formula", "Edit_dist"), 
                                     ("table", "TEDS"), 
                                     ("table", "TEDS_structure_only"), 
                                     ("reading_order", "Edit_dist")]:
            try:
                if metric == 'CDM' or metric == "TEDS" or metric == "TEDS_structure_only":
                    if result.get(category_type, {}).get("page", {}).get(metric):
                        save_dict[category_type+'_'+metric] = result[category_type]["page"][metric]["ALL"] * 100
                    else:
                        save_dict[category_type+'_'+metric] = np.nan
                else:
                    all_metric = result.get(category_type, {}).get("all", {}).get(metric, {})
                    save_dict[category_type+'_'+metric] = all_metric.get("ALL_page_avg", np.nan)
            except Exception as e:
                print(f"Error processing {category_type}_{metric}: {e}")
                save_dict[category_type+'_'+metric] = np.nan
        
        dict_list.append(save_dict)
        ocr_names.append(ocr_type)
    
    if not dict_list:
        print("Error: No valid result files found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(dict_list, index=ocr_names).round(3)
    
    # Calculate overall score
    df['overall'] = ((1 - df['text_block_Edit_dist']/100) * 100 + 
                     (1 - df['display_formula_Edit_dist']/100) * 100 + 
                     df['table_TEDS']) / 3
    
    return df


if __name__ == '__main__':
    parameters = process_args(sys.argv[1:])
    
    print("="*80)
    print("OmniDocBench Evaluation Results Summary")
    print("="*80)
    print(f"Result Directory: {parameters.result_dir}")
    print(f"OCR Types: {', '.join(parameters.ocr_types)}")
    print(f"Language Filter: {parameters.language}")
    print(f"Match Method: {parameters.match_method}")
    print("="*80)
    
    df = load_and_process_results(parameters.result_dir, 
                                   parameters.ocr_types,
                                   parameters.language, 
                                   parameters.match_method)
    
    if df is not None:
        print("\n")
        print(df.to_string())
        print("\n")
        
        # Save to CSV if specified
        if parameters.output:
            df.to_csv(parameters.output)
            print(f"Results saved to: {parameters.output}")
        
        print("="*80)

