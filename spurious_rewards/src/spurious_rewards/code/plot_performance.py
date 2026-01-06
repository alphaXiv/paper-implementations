#!/usr/bin/env python3
"""
Plot performance metrics across checkpoints for different datasets.

Usage: python plot_performance.py [--base-model] step1 step2 step3 ...
"""

import json
import matplotlib.pyplot as plt
import os
import sys

BENCHMARKS = {
    'MATH-500': {'rollouts': 1},
    'AIME-2024': {'rollouts': 8},
    'AIME-2025': {'rollouts': 8},
    'AMC': {'rollouts': 8},
}

# Parse arguments
args = sys.argv[1:]
include_base = False

if '--base-model' in args:
    include_base = True
    args.remove('--base-model')

if len(args) < 1:
    print("Usage: python plot_performance.py [--base-model] step1 step2 step3 ...")
    sys.exit(1)

steps = [int(x) for x in args]
datasets = list(BENCHMARKS.keys())

for metric_base in ['avg', 'pass']:
    plt.figure(figsize=(12, 7))
    
    # Store base model values for baseline plotting
    base_values = {}
    
    # Load base model data if requested
    if include_base:
        for dataset in datasets:
            rollouts = BENCHMARKS[dataset]['rollouts']
            metric = f'{metric_base}@{rollouts}'
            
            # Try to load base model results - look for files with model name prefix
            base_dir = 'results/base'
            base_found = False
            
            if os.path.exists(base_dir):
                # List all files in base directory and find matching dataset
                for filename in os.listdir(base_dir):
                    if dataset in filename and filename.endswith('.json'):
                        base_file = os.path.join(base_dir, filename)
                        try:
                            with open(base_file) as f:
                                data = json.load(f)
                                base_values[dataset] = data.get(metric, 0)
                                print(f"Found base model {metric} for {dataset}: {base_values[dataset]} from {filename}")
                                base_found = True
                                break
                        except Exception as e:
                            print(f"Error reading base model file {base_file}: {e}")
            
            if not base_found:
                print(f"Base model file not found for dataset: {dataset}")
    
    for dataset in datasets:
        rollouts = BENCHMARKS[dataset]['rollouts']
        metric = f'{metric_base}@{rollouts}'
        values = []
        valid_steps = []
        
        for step in steps:
            # Try multiple possible filenames
            if step == 0:
                file_path = f'export for eval step0/export-for-eval-step0_{dataset}.json'
            else:
                file_path = f'results/step{step}/export-for-eval-step{step}_{dataset}.json'
            
            if os.path.exists(file_path):
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                        val = data.get(metric, 0)
                        values.append(val)
                        valid_steps.append(step)
                        print(f"Found {metric} for {dataset} at step {step}: {val}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                print(f"File not found: {file_path}")
        
        if valid_steps and values:
            plt.plot(valid_steps, values, marker='o', linewidth=2.5, markersize=8, label=f'{dataset} ({metric})')
            
        # Plot base model as horizontal line if available
        if include_base and dataset in base_values and valid_steps:
            plt.axhline(y=base_values[dataset], color=plt.gca().lines[-1].get_color(), 
                       linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'{dataset} Base Model')
    
    plt.xlabel('Checkpoint Step', fontsize=13, fontweight='bold')
    plt.ylabel(f'{metric_base}@k', fontsize=13, fontweight='bold')
    plt.title(f'{metric_base}@k across checkpoints for different datasets', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{metric_base}_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved as {metric_base}_performance.png')

print('All plots saved successfully!')