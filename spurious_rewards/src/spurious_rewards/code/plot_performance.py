#!/usr/bin/env python3
"""
Plot performance metrics across checkpoints for different datasets.

Usage: python plot_performance.py step1 step2 step3 ...
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

if len(sys.argv) < 2:
    print("Usage: python plot_performance.py step1 step2 step3 ...")
    sys.exit(1)

steps = [int(x) for x in sys.argv[1:]]
datasets = list(BENCHMARKS.keys())

for metric_base in ['avg', 'pass']:
    plt.figure(figsize=(12, 7))
    for dataset in datasets:
        rollouts = BENCHMARKS[dataset]['rollouts']
        metric = f'{metric_base}@{rollouts}'
        values = []
        valid_steps = []
        
        for step in steps:
            # Try multiple possible filenames
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