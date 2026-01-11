#!/usr/bin/env python3
"""
Visualize evaluation results for Wikitext-2 and SNLI tasks.
Creates bar graphs comparing Grassmann and Transformer models.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('default')
colors_transformer = '#2E86AB'
colors_grassmann = '#A23B72'

def create_wikitext_comparison():
    """Create bar graph comparing Wikitext-2 perplexity across configurations."""
    
    # Our reproduction results (TEST SET) - from outputs/2026-01-10_23-46-12
    configs = ['L128\nN6', 'L128\nN12', 'L256\nN6', 'L256\nN12']
    
    transformer_ppl_ours = [181.66, 170.43, 180.85, 168.68]
    grassmann_ppl_ours = [253.76, 244.61, 251.32, 245.10]
    
    # Paper's reported results (VALIDATION SET)
    # Table 1: L128 N6 - Transformer: 248.4, Grassmann: 275.7
    # Table 2: L256 N12 - Transformer: 235.2, Grassmann: 261.1
    transformer_ppl_paper = [248.4, None, None, 235.2]
    grassmann_ppl_paper = [275.7, None, None, 261.1]
    
    x = np.arange(len(configs))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Our results
    bars1 = ax.bar(x - width*1.5, transformer_ppl_ours, width, label='Transformer (Our Test)',
                   color=colors_transformer, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x - width/2, grassmann_ppl_ours, width, label='Grassmann (Our Test)',
                   color=colors_grassmann, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Paper's results (where available)
    bars3 = ax.bar(x + width/2, [v if v else 0 for v in transformer_ppl_paper], width, 
                   label='Transformer (Paper Val)',
                   color=colors_transformer, alpha=0.4, edgecolor='black', linewidth=1.2, hatch='//')
    bars4 = ax.bar(x + width*1.5, [v if v else 0 for v in grassmann_ppl_paper], width, 
                   label='Grassmann (Paper Val)',
                   color=colors_grassmann, alpha=0.4, edgecolor='black', linewidth=1.2, hatch='//')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add paper values where available
    for i, (bar, val) in enumerate(zip(bars3, transformer_ppl_paper)):
        if val:
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkblue')
    
    for i, (bar, val) in enumerate(zip(bars4, grassmann_ppl_paper)):
        if val:
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')
    
    # Styling
    ax.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Perplexity (lower is better)', fontsize=14, fontweight='bold')
    ax.set_title('Wikitext-2 Language Modeling: Our Reproduction vs Paper Claims\n(Our Test Results vs Paper Validation Results)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/wikitext_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: images/wikitext_results.png")
    
    return fig


def create_snli_comparison():
    """Create bar graph showing SNLI classification accuracy - ours vs paper."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Our results - Per-class comparison (from outputs/2026-01-10_23-46-12)
    categories = ['Overall', 'Entailment', 'Neutral', 'Contradiction']
    grassmann_acc = [71.25, 76.07, 66.85, 70.62]
    transformer_acc = [66.71, 74.41, 62.29, 63.11]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, transformer_acc, width, label='Transformer (Our Test)',
                    color=colors_transformer, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, grassmann_acc, width, label='Grassmann (Our Test)',
                    color=colors_grassmann, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Styling
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Our Results: SNLI Classification (Test Set)\nGrassmann outperforms Transformer (+4.54%)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylim([0, 80])
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Paper's results - Overall accuracy comparison
    models_paper = ['Transformer\nHead\n(Paper Test)', 'Grassmann\nHead\n(Paper Test)',
                    'Transformer\n(Our Test)', 'Grassmann\n(Our Test)']
    accuracies_paper = [85.11, 85.38, 66.71, 71.25]
    colors_paper = [colors_transformer, colors_grassmann, 
                    colors_transformer, colors_grassmann]
    
    bars3 = ax2.bar(range(len(models_paper)), accuracies_paper, color=colors_paper,
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add hatching for paper results
    bars3[0].set_hatch('//')
    bars3[0].set_alpha(0.4)
    bars3[1].set_hatch('//')
    bars3[1].set_alpha(0.4)
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax2.text(i, height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(range(len(models_paper)))
    ax2.set_xticklabels(models_paper, fontsize=10)
    ax2.set_ylabel('Overall Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Paper vs Our Results: SNLI Overall Accuracy\n(Paper uses DistilBERT backbone, ours from scratch)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylim([0, 95])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add gap annotations
    ax2.plot([0, 1], [85.11, 85.38], 'g--', linewidth=2, alpha=0.5, label='Paper Gap: +0.27%')
    ax2.plot([2, 3], [66.71, 71.25], 'b--', linewidth=2, alpha=0.5, label='Our Gap: +4.54%')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('images/snli_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: images/snli_results.png")
    
    return fig


def create_parameter_comparison():
    """Create comparison showing performance analysis."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Comparative bar chart - Best models only
    models = ['Transformer\nL256 N12\n(Our Test)', 'Grassmann\nL256 N12\n(Our Test)', 
              'Transformer\nL128 N6\n(Paper Val)', 'Grassmann\nL128 N6\n(Paper Val)',
              'Transformer\nL256 N12\n(Paper Val)', 'Grassmann\nL256 N12\n(Paper Val)']
    values = [168.68, 245.10, 248.4, 275.7, 235.2, 261.1]
    colors_bars = [colors_transformer, colors_grassmann, 
                   colors_transformer, colors_grassmann,
                   colors_transformer, colors_grassmann]
    alphas = [0.9, 0.9, 0.4, 0.4, 0.4, 0.4]
    
    bars = ax.bar(range(len(models)), values, color=colors_bars, 
                  edgecolor='black', linewidth=1.5)
    
    # Set individual alpha and hatching
    for i, bar in enumerate(bars):
        bar.set_alpha(alphas[i])
        if i >= 2:  # Paper results
            bar.set_hatch('//')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(i, height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Perplexity', fontsize=13, fontweight='bold')
    ax.set_title('Best Model Comparison: Our Reproduction vs Paper', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add gap lines
    ax.plot([0, 1], [168.68, 245.10], 'r--', linewidth=2, alpha=0.5, label='Our Gap: 45.3%')
    ax.plot([2, 3], [248.4, 275.7], 'orange', linestyle='--', linewidth=2, alpha=0.5, label='Paper Gap: 11.0%')
    ax.plot([4, 5], [235.2, 261.1], 'orange', linestyle='--', linewidth=2, alpha=0.5)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('images/parameter_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: images/parameter_comparison.png")
    
    return fig


def create_combined_dashboard():
    """Create a comprehensive dashboard with all results."""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Wikitext-2 Perplexity Comparison (Larger, top spanning)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Our reproduction results (TEST SET)
    configs = ['L128 N6', 'L128 N12', 'L256 N6', 'L256 N12']
    transformer_ppl_ours = [181.66, 170.43, 180.85, 168.68]
    grassmann_ppl_ours = [253.76, 244.61, 251.32, 245.10]
    
    # Paper's reported results (VALIDATION SET)
    transformer_ppl_paper = [248.4, None, None, 235.2]
    grassmann_ppl_paper = [275.7, None, None, 261.1]
    
    x = np.arange(len(configs))
    width = 0.2
    
    # Our results
    ax1.bar(x - width*1.5, transformer_ppl_ours, width, label='Transformer (Our Test)',
            color=colors_transformer, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.bar(x - width/2, grassmann_ppl_ours, width, label='Grassmann (Our Test)',
            color=colors_grassmann, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Paper's results (where available)
    bars3 = ax1.bar(x + width/2, [v if v else 0 for v in transformer_ppl_paper], width, 
                    label='Transformer (Paper Val)',
                    color=colors_transformer, alpha=0.4, edgecolor='black', linewidth=1.2, hatch='//')
    bars4 = ax1.bar(x + width*1.5, [v if v else 0 for v in grassmann_ppl_paper], width, 
                    label='Grassmann (Paper Val)',
                    color=colors_grassmann, alpha=0.4, edgecolor='black', linewidth=1.2, hatch='//')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.set_ylabel('Perplexity (lower is better)', fontweight='bold', fontsize=12)
    ax1.set_title('Wikitext-2: Our Test Results vs Paper Validation Results', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. SNLI Accuracy by Class
    ax2 = fig.add_subplot(gs[1, 0])
    categories = ['Overall', 'Entailment', 'Neutral', 'Contradiction']
    grassmann_acc = [66.36, 72.68, 61.17, 64.94]
    transformer_acc = [62.63, 71.44, 56.76, 59.31]
    
    x_snli = np.arange(len(categories))
    width_snli = 0.35
    
    ax2.bar(x_snli - width_snli/2, transformer_acc, width_snli, label='Transformer',
            color=colors_transformer, alpha=0.8, edgecolor='black')
    ax2.bar(x_snli + width_snli/2, grassmann_acc, width_snli, label='Grassmann',
            color=colors_grassmann, alpha=0.8, edgecolor='black')
    
    ax2.set_xticks(x_snli)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('SNLI: Our Results (Grassmann wins +3.73%)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 80])
    ax2.legend(fontsize=10)
    
    # 3. Performance Gap Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    gap_configs = ['L128 N6\n(Ours)', 'L256 N12\n(Ours)', 'L128 N6\n(Paper)', 'L256 N12\n(Paper)']
    gaps = [31.2, 43.5, 11.0, 11.0]
    gap_colors = ['#DC143C', '#DC143C', 'orange', 'orange']
    bars = ax3.bar(gap_configs, gaps, color=gap_colors, alpha=0.7, edgecolor='black')
    bars[2].set_hatch('//')
    bars[3].set_hatch('//')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('Performance Gap (%)', fontweight='bold')
    ax3.set_title('Grassmann vs Transformer Gap', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 50])
    
    fig.suptitle('Grassmann Flows: Complete Evaluation Results', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('images/complete_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: images/complete_dashboard.png")
    
    return fig


if __name__ == '__main__':
    print("Generating visualizations...")
    print()
    
    # Create individual plots
    create_wikitext_comparison()
    create_snli_comparison()
    
    print()
    print("=" * 60)
    print("All visualizations created successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  • images/wikitext_results.png - Wikitext-2 bar chart (with paper comparison)")
    print("  • images/snli_results.png - SNLI accuracy breakdown (with paper comparison)")
    print()
    print("To view: open images/<filename>.png")
