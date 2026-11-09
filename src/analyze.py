"""Per-class and per-image analysis of the baseline on the TEST split.

1. AP by tooth type at IoU 0.50 / 0.75 / 0.85 and 0.50:0.95 (mean ± SD over seeds), plus AP at
   every IoU threshold: does the model lose more at loose thresholds (missing teeth) or strict
   ones (imprecise boundaries and root tips)?
2. Per-image mean mask IoU distribution (histogram) and the easiest / hardest images.
3. Precision vs recall at the locked confidence, with what the false negatives and false
   positives are made of.

Example:
    python src/analyze.py --data data/teeth_yolo/data.yaml --data-root data/teeth_yolo \
        --runs runs/baseline_s0/weights/best.pt runs/baseline_s1/weights/best.pt --out results/analysis
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (ROOTS, TOOTH_TYPES, VAL_CONF, VAL_IOU, PREDICT_CONF, read_names, seed_of,  # noqa: E402
                        tooth_number, tooth_type, train_imgsz)
from src.evaluate import evaluate  # noqa: E402
from src.imgeval import counts, label_for, run_image, split_images  # noqa: E402

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated: dataviz validate_palette.js
MARKERS = ["o", "s", "^", "D", "v"]


def ms(v):
    return f"{st.mean(v):.3f} ± {(st.stdev(v) if len(v) > 1 else 0.0):.3f}"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def locked(path=None):
    import os
    p = Path(path or os.environ.get("POSTPROCESS_YAML", "configs/postprocess.yaml"))
    if p.exists():
        d = yaml.safe_load(p.read_text())
        return float(d["conf"]), float(d["iou"])
    return PREDICT_CONF, VAL_IOU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conf, iou = locked()
    names = read_names(args.data)
    types = list(TOOTH_TYPES)

    # ---- 1. AP by tooth type and IoU threshold
    per_type = {t: {th: [] for th in THRESHOLDS} for t in types + ["all teeth"]}
    for w in args.runs:
        m = evaluate(w, args.data, "test", VAL_CONF, iou)
        all_ap, idx = np.asarray(m.seg.all_ap), list(m.seg.ap_class_index)
        nums = [tooth_number(names[int(c)]) for c in idx]
        for t in types + ["all teeth"]:
            rows = [k for k, n in enumerate(nums) if t == "all teeth" or tooth_type(n) == t]
            for j, th in enumerate(THRESHOLDS):
                per_type[t][th].append(float(all_ap[rows, j].mean()) if rows else float("nan"))
    with (out / "ap_by_type.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["type", "roots", "iou_threshold", "mean_ap", "sd_ap", "seeds"])
        for t in per_type:
            for th in THRESHOLDS:
                v = per_type[t][th]
                wr.writerow([t, ROOTS.get(t, ""), th, round(st.mean(v), 4),
                             round(st.stdev(v), 4) if len(v) > 1 else 0, len(v)])
    lines = ["| Tooth type | Roots | AP@50 | AP@75 | AP@85 | AP@50:95 |", "| --- | --- | --- | --- | --- | --- |"]
    for t in types + ["all teeth"]:
        d = per_type[t]
        a5095 = [float(np.mean([d[th][s] for th in THRESHOLDS])) for s in range(len(args.runs))]
        lines.append(f"| {t} | {ROOTS.get(t, '')} | {ms(d[0.50])} | {ms(d[0.75])} | {ms(d[0.85])} | {ms(a5095)} |")
    drop = {t: st.mean(per_type[t][0.50]) - st.mean(per_type[t][0.85]) for t in types}
    lines += ["", f"Mean over {len(args.runs)} seeds, test split, NMS IoU {iou}. Largest AP@50 -> AP@85 drop: "
              f"{max(drop, key=drop.get)} ({max(drop.values()):.3f}); smallest: {min(drop, key=drop.get)} "
              f"({min(drop.values()):.3f})."]
    (out / "ap_by_type.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axp = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    style(axp)
    for k, t in enumerate(types):
        y = [st.mean(per_type[t][th]) for th in THRESHOLDS]
        axp.plot(THRESHOLDS, y, color=SERIES[k], lw=2, marker=MARKERS[k], ms=6, label=f"{t} ({ROOTS[t]})")
    axp.set_xlim(0.48, 0.97)
    axp.set_ylim(0, 1)
    axp.set_xlabel("IoU threshold for a correct mask", color=INK2, fontsize=10)
    axp.set_ylabel("Mask AP (test, mean over seeds)", color=INK2, fontsize=10)
    axp.set_title("AP falls as the IoU threshold tightens", color=INK, fontsize=12, loc="left")
    axp.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), labelcolor=INK2,
               title="Tooth type", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "ap_vs_iou_by_type.png", dpi=160)
    plt.close(fig)

    # ---- 2 + 3. per-image IoU and precision/recall at the locked confidence
    images = split_images(args.data_root, "test")
    per_image, pr_rows, fn_all, fp_all = {}, [], [], []
    for w in args.runs:
        model, seed, imgsz = YOLO(w), seed_of(w), train_imgsz(w)
        results = []
        for p in images:
            r = run_image(model, p, label_for(p), names, conf, iou, imgsz)
            results.append({"image": r["image"], "score": r["score"], "gt": r["gt"], "pred": r["pred"]})
            per_image.setdefault(r["image"], []).append(r["score"])
        c, fp_k, fn_k = counts(results)
        pr_rows.append({"seed": seed, **{k: c[k] for k in ("tp", "fp", "fn", "gt", "pred")},
                        "precision": round(c["precision"], 4), "recall": round(c["recall"], 4), "f1": round(c["f1"], 4)})
        fn_all.append(fn_k)
        fp_all.append(fp_k)

    with (out / "per_image_iou.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "mean_iou_over_seeds"] + [f"seed_{seed_of(w)}" for w in args.runs])
        for img, v in sorted(per_image.items(), key=lambda kv: np.mean(kv[1])):
            wr.writerow([img, round(float(np.mean(v)), 4)] + [round(x, 4) for x in v])
    scores = np.array([np.mean(v) for v in per_image.values()])
    q = np.percentile(scores, [10, 25, 50, 75, 90])
    ranked = sorted(per_image.items(), key=lambda kv: np.mean(kv[1]))

    fig, axh = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(axh)
    axh.hist(scores, bins=np.linspace(0, 1, 26), color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    axh.axvline(q[2], color=INK2, lw=1.2, ls="--")
    axh.annotate(f"median {q[2]:.2f}", (q[2], axh.get_ylim()[1] * 0.92), xytext=(-6, 0),
                 textcoords="offset points", ha="right", fontsize=9, color=INK2)
    axh.set_xlabel("Per-image mean mask IoU (each true tooth's best same-number match; missed = 0)",
                   color=INK2, fontsize=9)
    axh.set_ylabel("Test images", color=INK2, fontsize=10)
    axh.set_title(f"Most test X-rays score high; a tail of {int((scores < q[0] + 1e-9).sum())} hard cases sits "
                  f"below {q[0]:.2f}", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "per_image_iou_hist.png", dpi=160)
    plt.close(fig)

    with (out / "precision_recall.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(pr_rows[0]))
        wr.writeheader()
        wr.writerows(pr_rows)
    fn_tot = {k: sum(d.get(k, 0) for d in fn_all) / len(fn_all) for k in set().union(*fn_all)}
    fp_tot = {k: sum(d.get(k, 0) for d in fp_all) / len(fp_all) for k in set().union(*fp_all)}
    P, R = [r["precision"] for r in pr_rows], [r["recall"] for r in pr_rows]
    FN, FP = [r["fn"] for r in pr_rows], [r["fp"] for r in pr_rows]
    verdict = "false negatives (missed or mis-segmented teeth)" if st.mean(FN) > st.mean(FP) else \
        "false positives (extra, duplicate or wrongly numbered masks)"
    rep = [f"## Per-image mean mask IoU (test, {len(scores)} images, mean over seeds)", "",
           f"10th / 25th / 50th / 75th / 90th percentile: " + " / ".join(f"{x:.3f}" for x in q),
           f"Min {scores.min():.3f}, max {scores.max():.3f}, {int((scores < 0.5).sum())} images below 0.5.", "",
           "Hardest: " + ", ".join(f"{k} ({np.mean(v):.2f})" for k, v in ranked[:5]),
           "Easiest: " + ", ".join(f"{k} ({np.mean(v):.2f})" for k, v in ranked[-5:]), "",
           f"## Precision vs recall at the locked operating point (conf {conf}, NMS IoU {iou}; match = same tooth number and IoU >= 0.5)", "",
           "| Seed | TP | FP | FN | Precision | Recall | F1 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    rep += [f"| {r['seed']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
            for r in pr_rows]
    rep += [f"| mean ± SD | | {ms(FP)} | {ms(FN)} | {ms(P)} | {ms(R)} | {ms([r['f1'] for r in pr_rows])} |", "",
            f"Errors are dominated by **{verdict}**.", "",
            "False negatives per seed, by cause: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(fn_tot.items(), key=lambda kv: -kv[1])),
            "False positives per seed, by kind: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(fp_tot.items(), key=lambda kv: -kv[1]))]
    (out / "per_image_and_pr.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
