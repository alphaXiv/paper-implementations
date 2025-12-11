"""

Elo ratings for olmOCR vs baselines.

See data at scripts/elo/ratings.csv

    MethodA,MethodB,A_wins,B_wins,A_rate(%),B_rate(%)
    marker,mineru,53,26,67.1,32.9
    mineru,pdelf,22,55,28.6,71.4
    gotocr_format,marker,26,45,36.6,63.4
    marker,pdelf,31,49,38.8,61.3
    gotocr_format,pdelf,29,41,41.4,58.6
    gotocr_format,mineru,38,37,50.7,49.3

Invoke via
    python calculate_elo_ratings.py ratings.csv --num-bootstrap 5000 --num-elo-sims 100 --confidence-level 95 --seed 123

Output:

    Bootstrapped Elo Ratings (95% CI):
    --------------------------------------------------
    pdelf        1813.0 ± 84.9 [1605.9, 1930.0]
    mineru       1545.2 ± 99.7 [1336.7, 1714.1]
    marker       1429.1 ± 100.7 [1267.6, 1645.5]
    gotocr_format 1212.7 ± 82.0 [1097.3, 1408.3]

    Pairwise Significance Tests:
    --------------------------------------------------
    gotocr_format vs marker       Δ = -216.3 [-470.8,  135.0] p = 0.218
    gotocr_format vs mineru       Δ = -332.5 [-567.5,   19.3] p = 0.051
    gotocr_format vs pdelf        Δ = -600.3 [-826.1, -344.3] p = 0.000*
    marker       vs mineru       Δ = -116.1 [-365.4,  246.5] p = 0.430
    marker       vs pdelf        Δ = -383.9 [-610.6,  -10.9] p = 0.044*
    mineru       vs pdelf        Δ = -267.8 [-517.3,  104.0] p = 0.135


@kylel

"""

import random
from itertools import combinations

import click
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_elo(matches_data, all_methods, k_factor=32, initial_rating=1500, n_replications=10, random_state=None):
    """Calculate Elo ratings with multiple replications per dataset"""
    all_ratings = {method: [] for method in all_methods}

    for _ in range(n_replications):
        matches = matches_data.sample(frac=1, replace=False, random_state=random_state).reset_index(drop=True)
        ratings = {method: initial_rating for method in all_methods}

        for _, row in matches.iterrows():
            method_a, method_b = row["MethodA"], row["MethodB"]
            a_wins, b_wins = row["A_wins"], row["B_wins"]

            for _ in range(int(a_wins)):
                ra, rb = update_single_match(ratings[method_a], ratings[method_b], 1, k_factor)
                ratings[method_a], ratings[method_b] = ra, rb

            for _ in range(int(b_wins)):
                ra, rb = update_single_match(ratings[method_a], ratings[method_b], 0, k_factor)
                ratings[method_a], ratings[method_b] = ra, rb

        for method in all_methods:
            all_ratings[method].append(ratings[method])

    return {method: np.mean(ratings) for method, ratings in all_ratings.items()}


def update_single_match(rating_a, rating_b, actual_score, k_factor):
    """Update ratings for a single match"""
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_rating_a = rating_a + k_factor * (actual_score - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - actual_score) - (1 - expected_a))
    return new_rating_a, new_rating_b


def bootstrap_elo_and_tests(df, num_bootstrap=1000, num_elo_sims=10, confidence_level=95, k_factor=32, initial_rating=1500, random_state=None):
    """Calculate bootstrapped Elo ratings with confidence intervals and perform pairwise significance tests"""

    ci_lower = (100 - confidence_level) / 2
    ci_upper = 100 - ci_lower

    all_methods = set(df["MethodA"].unique()) | set(df["MethodB"].unique())
    bootstrap_ratings = {method: [] for method in all_methods}

    for _ in tqdm(range(num_bootstrap)):
        bootstrap_sample = df.sample(n=len(df), replace=True, random_state=random_state)
        ratings = calculate_elo(bootstrap_sample, all_methods, k_factor, initial_rating, num_elo_sims)

        for method in all_methods:
            bootstrap_ratings[method].append(ratings[method])

    # Calculate statistics and perform significance tests
    results = {}

    # Basic statistics
    for method in all_methods:
        ratings_array = np.array(bootstrap_ratings[method])
        results[method] = {
            "mean": np.mean(ratings_array),
            "std": np.std(ratings_array),
            "ci_lower": np.percentile(ratings_array, ci_lower),
            "ci_upper": np.percentile(ratings_array, ci_upper),
            "bootstrap_samples": ratings_array,  # Store for significance testing
        }

    # Pairwise significance tests
    significance_tests = {}
    for method1, method2 in combinations(all_methods, 2):
        # Calculate difference distribution
        diff_distribution = results[method1]["bootstrap_samples"] - results[method2]["bootstrap_samples"]

        # Calculate p-value (two-tailed test)
        p_value = 2 * min(np.mean(diff_distribution >= 0), np.mean(diff_distribution <= 0))

        # Store results
        significance_tests[(method1, method2)] = {
            "diff_mean": np.mean(diff_distribution),
            "diff_ci_lower": np.percentile(diff_distribution, ci_lower),
            "diff_ci_upper": np.percentile(diff_distribution, ci_upper),
            "p_value": p_value,
        }

    return results, significance_tests


@click.command()
@click.argument("ratings_file", type=click.Path(exists=True))
@click.option("--num-bootstrap", default=1000, help="Number of bootstrap iterations")
@click.option("--num-elo-sims", default=10, help="Number of ELO simulations per bootstrap")
@click.option("--confidence-level", default=95, help="Confidence level for intervals (in percent)")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def main(ratings_file, num_bootstrap, num_elo_sims, confidence_level, seed):
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    df = pd.read_csv(ratings_file)

    # Calculate bootstrapped Elo ratings
    results, significance_tests = bootstrap_elo_and_tests(df, num_bootstrap=num_bootstrap, num_elo_sims=num_elo_sims)

    # Sort and display results
    print(f"\nBootstrapped Elo Ratings ({confidence_level}% CI):")
    print("-" * 50)
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True))
    for method, stats in sorted_results.items():
        print(f"{method:12} {stats['mean']:6.1f} ± {stats['std']:4.1f} [{stats['ci_lower']:6.1f}, {stats['ci_upper']:6.1f}]")

    # Display pairwise significance tests
    print("\nPairwise Significance Tests:")
    print("-" * 50)
    for (method1, method2), stats in significance_tests.items():
        sig_marker = "*" if stats["p_value"] < (1 - confidence_level / 100) else " "
        print(
            f"{method1:12} vs {method2:12} Δ = {stats['diff_mean']:6.1f} "
            + f"[{stats['diff_ci_lower']:6.1f}, {stats['diff_ci_upper']:6.1f}] "
            + f"p = {stats['p_value']:.3f}{sig_marker}"
        )


if __name__ == "__main__":
    main()
