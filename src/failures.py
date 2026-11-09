"""Failure taxonomy on the TEST split: the 20 worst images, a visual catalog, and mAP impact.

Every true tooth that is not matched (same number, mask IoU >= 0.5) gets ONE primary category,
checked in this order (cause-specific first; src/imgeval.py: categorize):
  edge clipping            the tooth touches the edge of the X-ray's field of view
  metallic / restoration   >= 15% of the tooth is metal-bright (fillings, crowns, implants, brackets)
  merged neighbours        one predicted mask covers >= 30% of this tooth and of a neighbour
  missed                   no predicted mask of any number overlaps it (IoU < 0.3)
  wrong number             found (IoU >= 0.5) but labelled as another tooth (mirror / neighbour / other)
  poor boundary            the right area but a loose or partial mask
False positives are counted too: phantom masks on bone, sinus or background ("other").
The metal and edge flags are image heuristics: check them in the gallery and correct
failures.csv (manual_notes) before reporting.

Impact: for each of the two most common categories, re-score the test set without the images
that category dominates (>= 2 of its teeth and >= half the image's failures). The mAP gain is a
rough estimate of what that failure costs (removing an image also removes its good teeth).

Example:
    python src/failures.py --data data/teeth_yolo/data.yaml --data-root data/teeth_yolo \
        --runs runs/baseline_s0/weights/best.pt runs/baseline_s1/weights/best.pt --out results/failures
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import VAL_CONF, read_names, seed_of, subset_yaml, train_imgsz  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.analyze import locked  # noqa: E402
from src.imgeval import label_for, run_image, split_images  # noqa: E402

CATS = ["edge clipping", "metallic / restoration", "merged neighbours", "missed",
        "wrong number (mirror)", "wrong number (neighbour)", "wrong number (other)", "poor boundary"]
TAG = {"edge clipping": "E", "metallic / restoration": "Mt", "merged neighbours": "Mg", "missed": "Ms",
       "wrong number (mirror)": "Wm", "wrong number (neighbour)": "Wn", "wrong number (other)": "Wo",
       "poor boundary": "B", "phantom": "FP"}
HEX = {"edge clipping": "#2a78d6", "metallic / restoration": "#eb6834", "merged neighbours": "#1baf7a",
       "missed": "#eda100", "wrong number (mirror)": "#e87ba4", "wrong number (neighbour)": "#e87ba4",
       "wrong number (other)": "#e87ba4", "poor boundary": "#6250d6", "phantom": "#e34948"}
BGR = {k: tuple(int(v[i:i + 2], 16) for i in (5, 3, 1)) for k, v in HEX.items()}


def outline(img, mask, color, t=2):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, t)


def draw(r, k=2):
    """Full-image overlay at k x the model frame so tags stay readable, with a colour key."""
    up = lambda m: cv2.resize(m.astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST).astype(bool)  # noqa: E731
    out = cv2.resize(r["lb"], None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    gms, pms = [up(m) for m in r["gm"]], [up(m) for m in r["pm"]]
    rng = np.random.default_rng(0)
    for m in pms:
        out[m] = (0.6 * out[m] + 0.4 * rng.integers(60, 255, 3)).astype(np.uint8)
    for m, g in zip(gms, r["gt"]):
        outline(out, m, (0, 200, 0) if g["ok"] else BGR[g["category"]], 1 if g["ok"] else 3)
    for m, g in zip(gms, r["gt"]):
        if not g["ok"]:
            ys, xs = np.nonzero(m)
            cv2.putText(out, f"{g['tooth']}{TAG[g['category']]}", (int(xs.min()), max(int(ys.min()) - 4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, BGR[g["category"]], 2)
    for m, p in zip(pms, r["pred"]):
        if p["kind"] == "phantom":
            outline(out, m, BGR["phantom"], 3)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, f"{r['image']}  per-image mean IoU {r['score']:.2f}   green outline = correct tooth", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    items = []
    for cat, tag in TAG.items():
        if cat.startswith("wrong number (n") or cat.startswith("wrong number (o"):
            continue
        items.append((BGR[cat], {"wrong number (mirror)": "Wm/Wn/Wo = wrong number (mirror/neighbour/other)",
                                 "phantom": "FP = false positive"}.get(cat, f"{tag} = {cat}")))
    rows, x, y = [[]], 8, 0
    for color, label in items:
        w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        if x + w > out.shape[1] - 8 and rows[-1]:
            rows.append([])
            x = 8
        rows[-1].append((x, color, label))
        x += w + 14
    key = np.zeros((24 * len(rows) + 6, out.shape[1], 3), np.uint8)
    for r_i, row in enumerate(rows):
        for x, color, label in row:
            cv2.putText(key, label, (x, 20 + 24 * r_i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    return np.vstack([out, key])


def crop_pair(r, i, pad=24):
    m = r["gm"][i]
    ys, xs = np.nonzero(m)
    S = m.shape[0]
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, S)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, S)
    gt = r["lb"].copy()
    outline(gt, m, (0, 200, 0), 2)
    pr = r["lb"].copy()
    for pm in r["pm"]:
        if (pm & m).any():
            pr[pm] = (0.55 * pr[pm] + 0.45 * np.array([214, 120, 42])).astype(np.uint8)
            outline(pr, pm, (214, 120, 42), 1)
    a, b = gt[y0:y1, x0:x1], pr[y0:y1, x0:x1]
    h = 220
    a = cv2.resize(a, (max(1, int(a.shape[1] * h / a.shape[0])), h))
    b = cv2.resize(b, (a.shape[1], h))
    tile = np.hstack([a, np.full((h, 4, 3), 255, np.uint8), b])
    label = f"{r['image']} tooth {r['gt'][i]['tooth']}: true (left) vs predicted (right)"
    bar = np.zeros((20, tile.shape[1], 3), np.uint8)
    cv2.putText(bar, label[:60], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return np.vstack([bar, tile])


def grid(tiles, cols, width):
    tiles = [cv2.resize(t, (width, int(t.shape[0] * width / t.shape[1]))) for t in tiles]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="baseline best.pt per seed; the first is analysed")
    ap.add_argument("--out", default="results/failures")
    ap.add_argument("--worst", type=int, default=20)
    args = ap.parse_args()

    from ultralytics import YOLO
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conf, iou = locked()
    names = read_names(args.data)
    w0 = args.runs[0]
    model, imgsz = YOLO(w0), train_imgsz(w0)
    images = split_images(args.data_root, "test")

    recs = []
    for p in images:
        r = run_image(model, p, label_for(p), names, conf, iou, imgsz)
        cats = [g["category"] for g in r["gt"] if not g["ok"]]
        recs.append({"path": p, "image": r["image"], "score": r["score"], "gt": r["gt"], "pred": r["pred"],
                     "cats": cats, "phantom": sum(x["kind"] == "phantom" for x in r["pred"]),
                     "duplicate": sum(x["kind"] == "duplicate" for x in r["pred"])})
    recs.sort(key=lambda r: r["score"])
    worst = recs[: args.worst]

    # ---- table per image and taxonomy counts
    fields = ["rank", "image", "mean_iou", "true_teeth", "failed_teeth"] + CATS + ["false_pos_phantom", "duplicates",
                                                                                     "dominant", "manual_notes"]
    rows = []
    for k, r in enumerate(worst, 1):
        c = {cat: r["cats"].count(cat) for cat in CATS}
        dom = max(CATS, key=lambda x: c[x]) if r["cats"] else ("false positives" if r["phantom"] else "none")
        rows.append({"rank": k, "image": r["image"], "mean_iou": round(r["score"], 3), "true_teeth": len(r["gt"]),
                     "failed_teeth": len(r["cats"]), **c, "false_pos_phantom": r["phantom"],
                     "duplicates": r["duplicate"], "dominant": dom, "manual_notes": ""})
    with (out / "failures_worst20.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def tally(rs):
        t = {cat: sum(r["cats"].count(cat) for r in rs) for cat in CATS}
        t["other: false positive (bone / sinus / background)"] = sum(r["phantom"] for r in rs)
        return t
    tw, ta = tally(worst), tally(recs)
    imgs_w = {cat: sum(cat in r["cats"] for r in worst) for cat in CATS}
    lines = [f"## Failure taxonomy: {len(worst)} worst test images (seed {seed_of(w0)}, conf {conf}, NMS IoU {iou})", "",
             f"| Failure type | Teeth (worst {len(worst)}) | Images affected (worst {len(worst)}) | Teeth (all {len(recs)} test) |", "| --- | --- | --- | --- |"]
    for cat in sorted(tw, key=lambda c: -tw[c]):
        lines.append(f"| {cat} | {tw[cat]} | {imgs_w.get(cat, sum(r['phantom'] > 0 for r in worst))} | {ta[cat]} |")
    lines += ["", "Metal and edge flags are heuristics: confirm in gallery_worst20.jpg and note corrections in "
              "failures_worst20.csv (manual_notes)."]

    # ---- visuals: re-run the worst images to draw them
    tiles, crops = [], {cat: [] for cat in CATS}
    for k, r0 in enumerate(worst, 1):
        r = run_image(model, r0["path"], label_for(r0["path"]), names, conf, iou, imgsz)
        vis = draw(r)
        cv2.imwrite(str(out / f"worst_{k:02d}_{Path(r['image']).stem}.jpg"), vis)
        tiles.append(vis)
        for i, g in enumerate(r["gt"]):
            if not g["ok"] and g["area"] > 0 and len(crops[g["category"]]) < 4:
                crops[g["category"]].append(crop_pair(r, i))
    cv2.imwrite(str(out / "gallery_worst20.jpg"), grid(tiles, 4, 640))
    top = sorted(CATS, key=lambda c: -tw[c])
    for cat in top[:4]:
        if crops[cat]:
            cv2.imwrite(str(out / f"catalog_{cat.split(' ')[0].replace('/', '')}.jpg"), grid(crops[cat], 2, 520))

    # ---- impact estimate for the two most common categories (all test images)
    top2 = [c for c in sorted(CATS, key=lambda c: -ta[c]) if ta[c] > 0][:2]
    lines += ["", f"## Estimated mAP impact of the top two failure types", "",
              "| Failure type | Images removed | mAP@50:95 all test | mAP@50:95 without them | Gain | Share of total mAP loss |",
              "| --- | --- | --- | --- | --- | --- |"]
    base = {w: float(evaluate(w, args.data, "test", VAL_CONF, iou).seg.map) for w in args.runs}
    for cat in top2:
        dom = [r["path"] for r in recs if r["cats"].count(cat) >= 2 and r["cats"].count(cat) >= len(r["cats"]) / 2]
        if not dom or len(dom) == len(recs):
            lines.append(f"| {cat} | {len(dom)} | | | | not estimable (no image dominated by it) |")
            continue
        keep = [r["path"] for r in recs if r["path"] not in dom]
        y = subset_yaml(args.data, keep, out / f"subset_without_{TAG[cat]}.yaml")
        gains, shares, b_all, b_wo = [], [], [], []
        for w in args.runs:
            v = float(evaluate(w, y, "test", VAL_CONF, iou).seg.map)
            b_all.append(base[w])
            b_wo.append(v)
            gains.append(v - base[w])
            shares.append((v - base[w]) / (1 - base[w]))
        sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0  # noqa: E731
        lines.append(f"| {cat} | {len(dom)} of {len(recs)} | {st.mean(b_all):.3f} | {st.mean(b_wo):.3f} | "
                     f"{st.mean(gains):+.3f} ± {sd(gains):.3f} | {100 * st.mean(shares):.0f}% |")
    lines += ["", "Gain = mAP without those images minus mAP on all test images (mean ± SD over seeds); share = gain / "
              "(1 - mAP). Rough: removing an image also removes its correctly segmented teeth."]
    (out / "failures.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
