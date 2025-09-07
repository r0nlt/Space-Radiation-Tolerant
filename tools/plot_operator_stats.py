#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict


def read_operator_stats(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Coerce types
            rows.append(
                {
                    "generation": int(r["generation"]),
                    "operator": r["operator"],
                    "applications": int(float(r["applications"])),
                    "success_rate": float(r["success_rate"]),
                    "credit_score": float(r["credit_score"]),
                    "probability": float(r.get("probability", 0.0)),
                    "diversity": float(r.get("diversity", 0.0)),
                    "adaptive_rate": float(r.get("adaptive_rate", 0.0)),
                }
            )
    return rows


def pivot_by_operator(rows, field):
    # returns: {operator: [(gen, value), ...]}
    series = defaultdict(list)
    for r in rows:
        series[r["operator"]].append((r["generation"], r[field]))
    # sort by generation
    for op in series:
        series[op].sort(key=lambda x: x[0])
    return series


def plot_lines(series_dict, title, ylabel, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available: {}".format(e))
        print("Skipping plot:", out_path)
        return

    plt.figure(figsize=(8, 5))
    for op, pts in series_dict.items():
        if not pts:
            continue
        xs = [g for g, _ in pts]
        ys = [v for _, v in pts]
        plt.plot(xs, ys, marker="o", label=op)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_diversity_adaptive(rows, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available: {}".format(e))
        print("Skipping plot:", out_path)
        return

    # group by generation (one row per operator; take first for diversity/rate)
    gen_map = {}
    for r in rows:
        g = r["generation"]
        if g not in gen_map:
            gen_map[g] = (r["diversity"], r["adaptive_rate"])

    gens = sorted(gen_map.keys())
    diversity = [gen_map[g][0] for g in gens]
    rate = [gen_map[g][1] for g in gens]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_title("Diversity and Adaptive Mutation Rate")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Diversity", color="tab:blue")
    ax1.plot(gens, diversity, marker="o", color="tab:blue", label="Diversity")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Adaptive Rate", color="tab:orange")
    ax2.plot(gens, rate, marker="s", color="tab:orange", label="Adaptive Rate")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot operator stats over generations")
    parser.add_argument("--csv", required=True, help="Path to operator_stats.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    rows = read_operator_stats(args.csv)
    if not rows:
        print("No rows found in:", args.csv)
        return

    os.makedirs(args.outdir, exist_ok=True)

    # Per-operator time series
    for field, title, ylabel, fname in [
        ("probability", "Operator Selection Probability", "Probability", "operator_probabilities.png"),
        ("credit_score", "Operator Credit Score", "Credit Score", "operator_credits.png"),
        ("success_rate", "Operator Success Rate", "Success Rate", "operator_success.png"),
    ]:
        series = pivot_by_operator(rows, field)
        plot_lines(series, title, ylabel, os.path.join(args.outdir, fname))

    # Diversity + adaptive rate combo plot
    plot_diversity_adaptive(rows, os.path.join(args.outdir, "diversity_adaptive_rate.png"))

    print("Saved plots to:", args.outdir)


if __name__ == "__main__":
    main()
