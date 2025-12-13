#!/usr/bin/env python3
"""
Simple CI calculator for a single sample proportion.

By default this script computes the Wald (normal-approx) 95% CI:

    p_hat +/- z_{1-alpha/2} * sqrt(p_hat*(1-p_hat)/n)

You can also request the Wilson interval via --method wilson.

Usage examples:
  python scripts/ci_wilson.py --n 1355 --p 84.03
  python scripts/ci_wilson.py --n 1355 --p 84.03 --method wilson
"""

import argparse
import math


def wilson_ci(p_hat, n, z=1.959963984540054):
    # convert proportion to counts
    x = p_hat * n
    phat = x / n
    z2 = z * z
    denom = 1 + z2 / n
    center = phat + z2 / (2 * n)
    sq = math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    lower = (center - z * sq) / denom
    upper = (center + z * sq) / denom
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return lower, upper


def wald_ci(p_hat, n, z=1.959963984540054):
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    me = z * se
    lower = p_hat - me
    upper = p_hat + me
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return lower, upper, se, me


def parse_args():
    parser = argparse.ArgumentParser(description='Compute CI for a single sample proportion')
    parser.add_argument('--n', type=int, required=True, help='sample size')
    parser.add_argument('--p', type=float, required=True, help='proportion in percent (e.g. 84.03)')
    parser.add_argument('--alpha', type=float, default=0.05, help='alpha for (1-alpha) CI (default 0.05)')
    parser.add_argument('--method', choices=['wald', 'wilson'], default='wald', help='CI method: wald (default) or wilson')
    return parser.parse_args()


def main():
    args = parse_args()
    n = args.n
    p_percent = args.p
    alpha = args.alpha
    method = args.method

    p_hat = p_percent / 100.0
    # Use fixed z approx as requested (approx 1.95)
    z = 1.95

    print(f"n = {n}")
    print(f"Observed proportion (input): {p_percent:.5f} %  (p_hat = {p_hat:.6f})")

    if method == 'wald':
        lower, upper, se, me = wald_ci(p_hat, n, z=z)
        print(f"Standard error (SE) = sqrt(p*(1-p)/n) = {se:.6f}")
        print(f"Margin of error (ME) = z * SE = {me:.6f}")
        print(f"Wald {100*(1-alpha):.1f}% CI: [{lower*100:.4f} %, {upper*100:.4f} %]")
    else:
        lower, upper = wilson_ci(p_hat, n, z=z)
        print(f"Wilson {100*(1-alpha):.1f}% CI: [{lower*100:.4f} %, {upper*100:.4f} %]")


if __name__ == '__main__':
    main()
