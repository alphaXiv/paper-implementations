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
    plt.figure(figsize=(10, 6))
    for dataset in datasets:
        rollouts = BENCHMARKS[dataset]['rollouts']
        metric = f'{metric_base}@{rollouts}'
        values = []
        for step in steps:
            file_path = f'results/step{step}/{dataset}.json'
            if os.path.exists(file_path):
                with open(file_path) as f:
                    data = json.load(f)
                    values.append(data.get(metric, 0))
            else:
                values.append(0)
        plt.plot(steps, values, marker='o', label=dataset)
    
    plt.xlabel('Checkpoint Step')
    plt.ylabel(f'{metric_base}@k')
    plt.title(f'{metric_base}@k across checkpoints for different datasets')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_base}_performance.png')
    plt.close()

print('Plots saved as avg_performance.png and pass_performance.png')