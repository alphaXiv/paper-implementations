#!/usr/bin/env python3
"""
Extract and analyze best validation results from training runs.
Compares with paper's reported results.
"""

import json
from pathlib import Path

def extract_results(base_dir="outputs/2026-01-09_13-33-09"):
    """Extract results from all model directories."""
    base_path = Path(base_dir)
    
    results = {
        "wikitext": {},
        "snli": {}
    }
    
    # Wikitext models
    for model_type in ["transformer", "grassmann"]:
        for block_size in [128, 256]:
            for num_layers in [6, 12]:
                model_dir = base_path / f"{model_type}_wikitext_L{block_size}_N{num_layers}"
                results_file = model_dir / "results.json"
                
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        
                    key = f"L{block_size}_N{num_layers}"
                    if model_type not in results["wikitext"]:
                        results["wikitext"][model_type] = {}
                    
                    results["wikitext"][model_type][key] = {
                        "params_M": round(data[model_type]["num_params"] / 1e6, 2),
                        "best_val_ppl": round(data[model_type].get("best_val_ppl", data[model_type].get("test_ppl", 0)), 2),
                        "best_epoch": data[model_type].get("best_epoch", 20),
                        "test_ppl": round(data[model_type]["test_ppl"], 2),
                    }
    
    # SNLI models
    for model_type in ["transformer", "grassmann"]:
        model_dir = base_path / f"{model_type}_snli"
        
        # Check for best checkpoint
        best_ckpt = model_dir / "checkpoints" / "best.pt"
        if best_ckpt.exists():
            import torch
            ckpt = torch.load(best_ckpt, map_location='cpu')
            
            results["snli"][model_type] = {
                "best_val_acc": round(ckpt.get("val_acc", 0) * 100, 2),
                "best_epoch": ckpt.get("epoch", 20),
            }
        
        # Get test results
        test_file = model_dir / "snli_test_results.json"
        if test_file.exists():
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            
            # Handle both old and new format
            test_acc = test_data.get("overall_accuracy")
            if test_acc is None:
                test_acc = test_data.get("accuracy", {}).get("mean", 0)
            
            if model_type in results["snli"]:
                results["snli"][model_type]["test_acc"] = round(test_acc * 100, 2)
            else:
                results["snli"][model_type] = {
                    "test_acc": round(test_acc * 100, 2)
                }
    
    return results


def create_comparison_table(results):
    """Create markdown comparison table with paper's results."""
    
    output = []
    
    # Wikitext-2 Comparison
    output.append("## 📊 Best Validation Results vs Paper")
    output.append("")
    output.append("### Wikitext-2 Language Modeling")
    output.append("")
    output.append("#### Our Results (Best Validation PPL)")
    output.append("")
    output.append("| Model | Config | Params | Best Val PPL | Best Epoch | Test PPL |")
    output.append("|-------|--------|--------|--------------|------------|----------|")
    
    for config in ["L128_N6", "L256_N12"]:
        if "transformer" in results["wikitext"] and config in results["wikitext"]["transformer"]:
            t = results["wikitext"]["transformer"][config]
            output.append(f"| Transformer | {config.replace('_', ' ')} | {t['params_M']}M | {t['best_val_ppl']} | {t['best_epoch']} | {t['test_ppl']} |")
        
        if "grassmann" in results["wikitext"] and config in results["wikitext"]["grassmann"]:
            g = results["wikitext"]["grassmann"][config]
            output.append(f"| Grassmann | {config.replace('_', ' ')} | {g['params_M']}M | {g['best_val_ppl']} | {g['best_epoch']} | {g['test_ppl']} |")
    
    output.append("")
    output.append("#### Paper's Reported Results (Best Validation PPL)")
    output.append("")
    output.append("| Model | Config | Params | Best Val PPL | Gap |")
    output.append("|-------|--------|--------|--------------|-----|")
    output.append("| Transformer | L128 N6 | 12.59M | 248.4 | baseline |")
    output.append("| Grassmann | L128 N6 | 13.00M | 275.7 | +11.0% |")
    output.append("| Transformer | L256 N12 | 17.32M | 235.2 | baseline |")
    output.append("| Grassmann | L256 N12 | 18.16M | 261.1 | +11.0% |")
    output.append("")
    
    # Calculate gaps
    if "transformer" in results["wikitext"] and "grassmann" in results["wikitext"]:
        output.append("#### Gap Comparison")
        output.append("")
        output.append("| Config | Our Val Gap | Paper Val Gap | Our Test Gap |")
        output.append("|--------|-------------|---------------|--------------|")
        
        for config, paper_t, paper_g in [("L128_N6", 248.4, 275.7), ("L256_N12", 235.2, 261.1)]:
            if config in results["wikitext"]["transformer"] and config in results["wikitext"]["grassmann"]:
                our_t = results["wikitext"]["transformer"][config]
                our_g = results["wikitext"]["grassmann"][config]
                
                our_val_gap = ((our_g["best_val_ppl"] / our_t["best_val_ppl"]) - 1) * 100
                paper_val_gap = ((paper_g / paper_t) - 1) * 100
                our_test_gap = ((our_g["test_ppl"] / our_t["test_ppl"]) - 1) * 100
                
                output.append(f"| {config.replace('_', ' ')} | {our_val_gap:+.1f}% | {paper_val_gap:+.1f}% | {our_test_gap:+.1f}% |")
        
        output.append("")
    
    # SNLI Comparison
    output.append("### SNLI Classification")
    output.append("")
    output.append("#### Our Results (Best Validation Accuracy)")
    output.append("")
    output.append("| Model | Best Val Acc | Best Epoch | Test Acc |")
    output.append("|-------|--------------|------------|----------|")
    
    for model_type in ["transformer", "grassmann"]:
        if model_type in results["snli"]:
            data = results["snli"][model_type]
            output.append(f"| {model_type.capitalize()} | {data.get('best_val_acc', 'N/A')}% | {data.get('best_epoch', 'N/A')} | {data.get('test_acc', 'N/A')}% |")
    
    output.append("")
    output.append("#### Paper's Reported Results (DistilBERT backbone)")
    output.append("")
    output.append("| Model | Val Acc | Test Acc |")
    output.append("|-------|---------|----------|")
    output.append("| Transformer head | 85.45% | 85.11% |")
    output.append("| Grassmann-Plücker head | 85.50% | 85.38% |")
    output.append("")
    output.append("**Note:** Paper uses pre-trained DistilBERT backbone. Our models trained from scratch.")
    output.append("")
    
    return "\n".join(output)


def save_json_summary(results, output_file="outputs/2026-01-09_13-33-09/best_results_summary.json"):
    """Save comprehensive JSON summary."""
    
    summary = {
        "metadata": {
            "run_date": "2026-01-09",
            "description": "Best validation results from reproduction study"
        },
        "wikitext2": {
            "our_results": results["wikitext"],
            "paper_results": {
                "transformer": {
                    "L128_N6": {"params_M": 12.59, "best_val_ppl": 248.4},
                    "L256_N12": {"params_M": 17.32, "best_val_ppl": 235.2}
                },
                "grassmann": {
                    "L128_N6": {"params_M": 13.00, "best_val_ppl": 275.7},
                    "L256_N12": {"params_M": 18.16, "best_val_ppl": 261.1}
                }
            }
        },
        "snli": {
            "our_results": results["snli"],
            "paper_results": {
                "transformer_head": {"val_acc": 85.45, "test_acc": 85.11},
                "grassmann_head": {"val_acc": 85.50, "test_acc": 85.38},
                "note": "Paper uses pre-trained DistilBERT backbone"
            }
        }
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ JSON summary saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Extracting best validation results...")
    results = extract_results()
    
    print("\nCreating comparison table...")
    table = create_comparison_table(results)
    print(table)
    
    print("\nSaving JSON summary...")
    json_file = save_json_summary(results)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
