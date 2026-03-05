#!/usr/bin/env python3
"""
Generate charts for BLOG.md tables
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Create assets directory if it doesn't exist
os.makedirs('assets', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'ES': '#FF6B6B', 'GRPO': '#4ECDC4'}

# ============================================================================
# Chart 1: Data Efficiency - ES vs GRPO on Instruction-Tuned Models
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Countdown data
countdown_data = {
    'Training %': [10, 40, 70, 100],
    'ES': [36.0, 35.0, 42.0, 39.0],
    'GRPO': [34.1, 39.6, 47.5, 40.5]
}

# GSM8K data
gsm8k_data = {
    'Training %': [10, 40, 70, 100],
    'ES': [89.0, 86.5, 83.0, 86.0],
    'GRPO': [85.5, 90.9, 89.6, 87.4]
}

# Plot Countdown
x = np.arange(len(countdown_data['Training %']))
width = 0.35
ax1.bar(x - width/2, countdown_data['ES'], width, label='ES', color=colors['ES'], alpha=0.8)
ax1.bar(x + width/2, countdown_data['GRPO'], width, label='GRPO', color=colors['GRPO'], alpha=0.8)
ax1.set_xlabel('Training Data Fraction (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Countdown Task', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['10%', '40%', '70%', '100%'])
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 62])  # Increased top margin
ax1.set_xlim([-0.6, 3.6])  # Add horizontal padding

# Add value labels on bars
for i, v in enumerate(countdown_data['ES']):
    ax1.text(i - width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(countdown_data['GRPO']):
    ax1.text(i + width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

# Plot GSM8K
ax2.bar(x - width/2, gsm8k_data['ES'], width, label='ES', color=colors['ES'], alpha=0.8)
ax2.bar(x + width/2, gsm8k_data['GRPO'], width, label='GRPO', color=colors['GRPO'], alpha=0.8)
ax2.set_xlabel('Training Data Fraction (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('GSM8K Task', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['10%', '40%', '70%', '100%'])
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([75, 97])  # Increased top margin
ax2.set_xlim([-0.6, 3.6])  # Add horizontal padding

# Add value labels on bars
for i, v in enumerate(gsm8k_data['ES']):
    ax2.text(i - width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(gsm8k_data['GRPO']):
    ax2.text(i + width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Data Efficiency: ES vs GRPO on Qwen2.5-3B-Instruct', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('assets/table1_data_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: assets/table1_data_efficiency.png")
plt.close()

# ============================================================================
# Chart 2: Base Models Comparison
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Base model data for Countdown (10% data, N=8, 100 iterations)
countdown_base = {
    'models': ['Qwen2.5-3B\n(ES)', 'Qwen2.5-3B\n(GRPO)', 'Llama-3.2-3B\n(ES)', 'Llama-3.2-3B\n(GRPO)'],
    'accuracy': [15.0, 58.43, 2.0, 23.46],
    'colors': [colors['ES'], colors['GRPO'], colors['ES'], colors['GRPO']]
}

# Base model data for GSM8K (10% data, N=8, 100 iterations)
gsm8k_base = {
    'models': ['Qwen2.5-3B\n(ES)', 'Qwen2.5-3B\n(GRPO)', 'Llama-3.2-3B\n(ES)', 'Llama-3.2-3B\n(GRPO)'],
    'accuracy': [82.5, 87.71, 16.0, 14.0],
    'colors': [colors['ES'], colors['GRPO'], colors['ES'], colors['GRPO']]
}

# Plot Countdown base models
bars1 = ax1.bar(countdown_base['models'], countdown_base['accuracy'], 
                color=countdown_base['colors'], alpha=0.8, width=0.6)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Countdown Task - Base Models', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 75])  # Increased top margin for better spacing
ax1.set_xlim([-0.6, 3.6])  # Add padding on sides

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, countdown_base['accuracy'])):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 2, 
             f'{val:.1f}%' if val >= 10 else f'{val:.0f}%', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot GSM8K base models
bars2 = ax2.bar(gsm8k_base['models'], gsm8k_base['accuracy'], 
                color=gsm8k_base['colors'], alpha=0.8, width=0.6)
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('GSM8K Task - Base Models', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 105])  # Increased top margin for better spacing
ax2.set_xlim([-0.6, 3.6])  # Add padding on sides

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, gsm8k_base['accuracy'])):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 2.5, 
             f'{val:.1f}%' if val >= 20 else f'{val:.0f}%', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add legend
legend_elements = [mpatches.Patch(color=colors['ES'], label='ES', alpha=0.8),
                   mpatches.Patch(color=colors['GRPO'], label='GRPO', alpha=0.8)]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
           fontsize=12, bbox_to_anchor=(0.5, 0.98))

plt.suptitle('Base Model Performance: ES vs GRPO', 
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('assets/table2_base_models.png', dpi=300, bbox_inches='tight')
print("✓ Saved: assets/table2_base_models.png")
plt.close()

# ============================================================================
# Chart 3: Population Size Scaling (N=8 vs N=30)
# ============================================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Qwen Countdown
qwen_countdown = {
    'N': ['N=8', 'N=30'],
    'accuracy': [36.0, 42.0]
}

# Qwen GSM8K
qwen_gsm8k = {
    'N': ['N=8', 'N=30'],
    'accuracy': [89.0, 87.5]
}

# Llama Countdown
llama_countdown = {
    'N': ['N=8', 'N=30'],
    'accuracy': [28.0, 38.0]
}

# Llama GSM8K
llama_gsm8k = {
    'N': ['N=8', 'N=30'],
    'accuracy': [82.0, 84.5]
}

pop_colors = ['#95E1D3', '#38A3A5']

# Uniform y-axis scaling for all subplots
COUNTDOWN_YLIM = [0, 52]
GSM8K_YLIM = [0, 100]

# Plot Qwen Countdown
bars = ax1.bar(qwen_countdown['N'], qwen_countdown['accuracy'], 
               color=pop_colors, alpha=0.8, width=0.5)
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Qwen2.5-3B-Instruct - Countdown', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(COUNTDOWN_YLIM)  # Uniform scaling
ax1.set_xlim([-0.5, 1.5])
for bar, val in zip(bars, qwen_countdown['accuracy']):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot Qwen GSM8K
bars = ax2.bar(qwen_gsm8k['N'], qwen_gsm8k['accuracy'], 
               color=pop_colors, alpha=0.8, width=0.5)
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Qwen2.5-3B-Instruct - GSM8K', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(GSM8K_YLIM)  # Uniform scaling
ax2.set_xlim([-0.5, 1.5])
for bar, val in zip(bars, qwen_gsm8k['accuracy']):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 2, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot Llama Countdown
bars = ax3.bar(llama_countdown['N'], llama_countdown['accuracy'], 
               color=pop_colors, alpha=0.8, width=0.5)
ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('Llama-3.2-3B-Instruct - Countdown', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(COUNTDOWN_YLIM)  # Uniform scaling
ax3.set_xlim([-0.5, 1.5])
for bar, val in zip(bars, llama_countdown['accuracy']):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot Llama GSM8K
bars = ax4.bar(llama_gsm8k['N'], llama_gsm8k['accuracy'], 
               color=pop_colors, alpha=0.8, width=0.5)
ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax4.set_title('Llama-3.2-3B-Instruct - GSM8K', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(GSM8K_YLIM)  # Uniform scaling
ax4.set_xlim([-0.5, 1.5])
for bar, val in zip(bars, llama_gsm8k['accuracy']):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 2, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Effect of Population Size (N=8 vs N=30) on ES Performance', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('assets/table3_population_scaling.png', dpi=300, bbox_inches='tight')
print("✓ Saved: assets/table3_population_scaling.png")
plt.close()

# ============================================================================
# Chart 4: Summary of Key Results
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

scenarios = [
    'Instruct models\n10% data',
    'Instruct models\n40-100% data',
    'Base models\n(Qwen)',
    'Base models\n(Llama)',
    'Large population\n(Countdown)',
    'Large population\n(GSM8K)'
]

winners = ['ES', 'GRPO', 'GRPO', 'Both fail', 'N=30', 'Mixed']
margins = ['Moderate\n(+3.5%)', 'Significant\n(+3-5%)', 'Moderate\n(+5.2%)', 
           'N/A\n(<25%)', 'Large\n(+6-10%)', 'Small\n(0-2%)']

# Color coding
winner_colors = []
for w in winners:
    if w == 'ES':
        winner_colors.append(colors['ES'])
    elif w == 'GRPO':
        winner_colors.append(colors['GRPO'])
    elif w == 'N=30':
        winner_colors.append('#95E1D3')
    elif w == 'Mixed':
        winner_colors.append('#FFA500')
    else:  # Both fail
        winner_colors.append('#999999')

y_pos = np.arange(len(scenarios))
bars = ax.barh(y_pos, [1]*len(scenarios), color=winner_colors, alpha=0.7)

# Add text annotations
for i, (scenario, winner, margin) in enumerate(zip(scenarios, winners, margins)):
    ax.text(0.05, i, f'{scenario}', va='center', fontsize=11, fontweight='bold')
    ax.text(0.5, i, f'Winner: {winner}', va='center', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.9, i, margin, va='center', ha='right', fontsize=10)

ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim([0, 1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add column headers
ax.text(0.05, len(scenarios) + 0.15, 'Scenario', fontsize=13, fontweight='bold', va='bottom')
ax.text(0.5, len(scenarios) + 0.15, 'Winner', fontsize=13, fontweight='bold', va='bottom', ha='center')
ax.text(0.9, len(scenarios) + 0.15, 'Margin', fontsize=13, fontweight='bold', va='bottom', ha='right')

# plt.title('Summary of Key Results:', fontsize=16, fontweight='bold', pad=30)
plt.tight_layout()
plt.savefig('assets/table4_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: assets/table4_summary.png")
plt.close()

print("\n✅ All charts generated successfully in assets/ directory!")
