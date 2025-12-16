# Evaluation Results Summary

The evaluation pipeline now automatically generates a results summary DataFrame and displays it at the end of each run.

## How It Works

After evaluation completes, the `run_evaluation.py` script is automatically called to:
1. Load metric results from the `result/` directory
2. Process results into a DataFrame with all key metrics
3. Calculate overall score
4. Display results in the terminal
5. Save results to CSV (optional)

## Metrics Included

The summary DataFrame includes:
- `text_block_Edit_dist` - Edit distance for text blocks
- `display_formula_Edit_dist` - Edit distance for display formulas
- `table_TEDS` - TEDS score for tables
- `table_TEDS_structure_only` - TEDS structure-only score for tables
- `reading_order_Edit_dist` - Edit distance for reading order
- `overall` - Calculated overall score (average of normalized metrics)

## Running the Pipeline

### With speedrun.sh (automatic results display):
```bash
# Run with all languages
bash speedrun.sh all

# Run with English filter
bash speedrun.sh english

# Run with Chinese filter
bash speedrun.sh simplified_chinese
```

The script will:
1. Set up conda environment
2. Configure language filter
3. Run evaluation
4. **Automatically display results summary**
5. Save CSV to `results_{language}.csv`

### Manual run_evaluation.py usage:

```bash
# Basic usage - show DeepSeek and OLMOCR2 results
python run_evaluation.py

# Specify custom parameters
python run_evaluation.py \
    --result-dir "./src/omnidocbench_evals/OmniDocBench/result" \
    --ocr-types deepseek olmocr2 \
    --language all \
    --match-method quick_match \
    --output results_all.csv

# English only
python run_evaluation.py \
    --result-dir "./src/omnidocbench_evals/OmniDocBench/result" \
    --ocr-types deepseek olmocr2 \
    --language english \
    --output results_english.csv

# Chinese only
python run_evaluation.py \
    --result-dir "./src/omnidocbench_evals/OmniDocBench/result" \
    --ocr-types deepseek olmocr2 \
    --language simplified_chinese \
    --output results_chinese.csv
```

## Command Line Arguments

```
--result-dir         Path to result directory (default: ./src/omnidocbench_evals/OmniDocBench/result)
--ocr-types          List of OCR types to process (default: deepseek olmocr2)
--language           Language filter: all, english, simplified_chinese (default: all)
--match-method       Match method: quick_match, no_split, etc. (default: quick_match)
--output             Output CSV file path (optional)
```

## Output Example

```
================================================================================
OmniDocBench Evaluation Results Summary
================================================================================
Result Directory: ./src/omnidocbench_evals/OmniDocBench/result
OCR Types: deepseek, olmocr2
Language Filter: all
Match Method: quick_match
================================================================================

         text_block_Edit_dist  display_formula_Edit_dist  table_TEDS  \
deepseek                 0.923                      0.945       0.915   
olmocr2                  0.931                      0.952       0.922   

         table_TEDS_structure_only  reading_order_Edit_dist  overall  
deepseek                       0.951                    0.088    88.234  
olmocr2                        0.958                    0.095    89.456  

================================================================================
Results saved to: results_all.csv
================================================================================
```

## Results Files

After evaluation, you'll find:

1. **Summary CSV**: `results_{language}.csv`
   - Contains the summary DataFrame
   - Ready for comparison and analysis

2. **Detailed JSON results** in `src/omnidocbench_evals/OmniDocBench/result/`:
   - `deepseek_{language}_quick_match_metric_result.json`
   - `olmocr2_{language}_quick_match_metric_result.json`
   - And corresponding per-page and per-sample results

## Workflow Example

```bash
# Step 1: Run evaluation on all languages
bash speedrun.sh all
# Output: results_all.csv is created and displayed

# Step 2: Run evaluation on English
bash speedrun.sh english
# Output: results_english.csv is created and displayed

# Step 3: Run evaluation on Chinese
bash speedrun.sh simplified_chinese
# Output: results_chinese.csv is created and displayed

# Step 4: Compare results
cat results_all.csv
cat results_english.csv
cat results_simplified_chinese.csv
```

## Notes

- Results are automatically displayed in the terminal at the end of each run
- CSV files are saved for further analysis
- The script handles missing result files gracefully with warnings
- All metrics are rounded to 3 decimal places
- The overall score is calculated as: (text_quality + formula_quality + table_quality) / 3
