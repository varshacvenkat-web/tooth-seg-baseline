"""Paired comparison across seeds: is a change bigger than seed noise?

Each CSV comes from src/evaluate.py (one row per seed). Rows are paired by seed, so both
setups share the same split, seeds and evaluation settings and differ in one thing only.

Reported per metric: mean ± SD of each setup, the exact per-seed deltas, a paired t-test,
a Wilcoxon signed-rank test (with only n seeds the smallest possible two-sided p is 2 / 2^n:
0.25 for 3 seeds, 0.0625 for 5, so it can never reach 0.05 with fewer than 6), and the
effect size Cohen's d_z = mean(delta) / SD(delta). The verdict also compares the mean
delta with the baseline's seed-to-seed SD.

Example:
    python src/compare.py --baseline results/baseline_test.csv \
        --other "Tuned post-processing=results/tuned_test.csv" "Exp 1=results/exp1_test.csv" --out results/comparison.md
"""
import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

METRICS = [("mask_mAP50", "Mask mAP@50"), ("mask_mAP50_95", "Mask mAP@50:95")]


def read(path):
    with open(path) as f:
        return {int(r["seed"]): {k: float(r[k]) for k, _ in METRICS} for r in csv.DictReader(f)}


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def paired_stats(a, b):
    """a, b: equal-length lists paired by seed. Returns dict of stats for b - a."""
    from scipy import stats
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    out = {"n": n, "deltas": d, "mean_delta": st.mean(d), "sd_delta": sd(d)}
    out["dz"] = out["mean_delta"] / out["sd_delta"] if out["sd_delta"] > 0 else (math.inf if out["mean_delta"] else 0.0)
    if n >= 2 and out["sd_delta"] > 0:
        t = stats.ttest_rel(b, a)
        out["t"], out["t_p"] = float(t.statistic), float(t.pvalue)
    else:
        out["t"], out["t_p"] = float("nan"), float("nan")
    if n >= 1 and any(x != 0 for x in d):
        w = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided", method="exact")
        out["w"], out["w_p"] = float(w.statistic), float(w.pvalue)
    else:
        out["w"], out["w_p"] = float("nan"), 1.0
    out["min_wilcoxon_p"] = 2 / 2 ** n
    return out


def verdict(s, base_sd):
    if abs(s["mean_delta"]) <= base_sd:
        return "within seed noise (|mean delta| <= baseline SD)"
    if not (s["t_p"] < 0.05):
        return "larger than the baseline SD but not significant (paired t p >= 0.05): treat as inconclusive"
    return ("real improvement" if s["mean_delta"] > 0 else "real degradation") + " (beyond seed SD, paired t p < 0.05)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--other", nargs="+", required=True, help='"Name=path.csv"')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = read(args.baseline)
    others = [(s.rsplit("=", 1)[0], read(s.rsplit("=", 1)[1])) for s in args.other]  # names may contain "="
    L = ["## Metric comparison (test split, paired by seed)", "",
         "| Setup | Seeds | " + " | ".join(f"{n} (mean ± SD)" for _, n in METRICS) + " |",
         "| --- | --- | " + " | ".join("---" for _ in METRICS) + " |"]
    L.append(f"| Baseline | {', '.join(map(str, sorted(base)))} | " + " | ".join(
        f"{st.mean([base[s][k] for s in base]):.4f} ± {sd([base[s][k] for s in base]):.4f}" for k, _ in METRICS) + " |")
    for name, o in others:
        L.append(f"| {name} | {', '.join(map(str, sorted(o)))} | " + " | ".join(
            f"{st.mean([o[s][k] for s in o]):.4f} ± {sd([o[s][k] for s in o]):.4f}" for k, _ in METRICS) + " |")

    for name, o in others:
        seeds = sorted(set(base) & set(o))
        missing = sorted(set(base) ^ set(o))
        L += ["", f"### {name} vs baseline (seeds {', '.join(map(str, seeds))})"]
        if missing:
            L.append(f"Unpaired seeds left out: {missing}.")
        L += ["", "| Metric | Per-seed delta (" + ", ".join(f"s{s}" for s in seeds) + ") | Mean delta ± SD | "
              "Paired t (p) | Wilcoxon W (p) | Cohen's d_z | Baseline seed SD | Verdict |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k, label in METRICS:
            a, b = [base[s][k] for s in seeds], [o[s][k] for s in seeds]
            s = paired_stats(a, b)
            base_sd = sd([base[x][k] for x in base])
            L.append(f"| {label} | " + ", ".join(f"{x:+.4f}" for x in s["deltas"]) +
                     f" | {s['mean_delta']:+.4f} ± {s['sd_delta']:.4f} | {s['t']:.2f} ({s['t_p']:.3f}) | "
                     f"{s['w']:.1f} ({s['w_p']:.3f}) | {s['dz']:.2f} | {base_sd:.4f} | {verdict(s, base_sd)} |")
        L.append(f"\nWith {len(seeds)} seeds the smallest possible Wilcoxon p is {2 / 2 ** len(seeds):.3f}, so it "
                 f"cannot show significance on its own; read it together with the paired t-test and effect size.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
