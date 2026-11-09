"""Split integrity (leakage) check and baseline configuration audit.

Leakage: the dataset export has no patient IDs, so patient-level grouping is impossible to
verify directly. As the closest check we confirm that
  1. no file name appears in more than one split,
  2. no byte-identical image appears in more than one split (MD5),
  3. no re-export of the same radiograph crosses splits. The dataset contains the same scan
     exported twice with different contrast processing (same burned-in date and time), so every
     pair of images is compared after histogram equalisation (which removes contrast differences):
       image similarity  = correlation of the equalised X-rays (text strip excluded), 256 x 128
       strip similarity  = correlation of the bottom text strip (scanner date/time/settings)
     A pair is a likely duplicate when image similarity >= 0.90 and strip similarity >= 0.80.
     Likely duplicates are joined into groups (duplicate_groups.json) that convert.py --groups
     keeps in one split. The 24 most similar cross-split pairs are saved side by side
     (near_duplicates.jpg) and all scores to near_duplicates.csv, to confirm by eye.
Config audit: reads each run's args.yaml and states input resolution, normalization, NMS,
optimizer, epochs and which split chose the checkpoint (it must be val, never test).

Example:
    python src/check_integrity.py --data-root data/teeth_yolo --runs runs/baseline_s0 runs/baseline_s1 \
        --out results/integrity.md
"""
import argparse
import csv
import hashlib
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PAD_VALUE  # noqa: E402

SPLITS = ("train", "val", "test")


def dhash(path, size=8) -> int:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def check_splits(root, out_dir=None):
    files = {s: sorted((Path(root) / "images" / s).iterdir()) for s in SPLITS}
    L, ok = [], True
    names = {s: {p.name for p in v} for s, v in files.items()}
    for a, b in itertools.combinations(SPLITS, 2):
        both = names[a] & names[b]
        ok &= not both
        L.append(f"- File names shared by {a} and {b}: {len(both)}" + (f" ({sorted(both)[:5]})" if both else ""))
    md5 = {}
    for s, v in files.items():
        for p in v:
            md5.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append((s, p.name))
    cross = [v for v in md5.values() if len({s for s, _ in v}) > 1]
    within = [v for v in md5.values() if len(v) > 1 and len({s for s, _ in v}) == 1]
    ok &= not cross
    L.append(f"- Byte-identical images across splits: {len(cross)}" + (f" {cross[:3]}" if cross else ""))
    L.append(f"- Byte-identical images within one split (no leakage, noted): {len(within)}")
    allp = [(s, p) for s, v in files.items() for p in v]
    F, T = [], []
    for _, p in allp:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h = g.shape[0]
        body = cv2.resize(cv2.equalizeHist(g[: int(0.93 * h)]), (256, 128), interpolation=cv2.INTER_AREA)
        strip = cv2.resize(g[int(0.94 * h):], (512, 16), interpolation=cv2.INTER_AREA)
        for arr, out in ((body, F), (strip, T)):
            a = arr.astype(np.float32).ravel()
            out.append((a - a.mean()) / (a.std() + 1e-6) / np.sqrt(a.size))
    F, T = np.stack(F), np.stack(T)
    C, S = F @ F.T, T @ T.T
    split_of = np.array([SPLITS.index(s) for s, _ in allp])
    iu = np.triu_indices(len(allp), 1)
    cross = split_of[iu[0]] != split_of[iu[1]]
    pairs = sorted(((float(C[i, j]), float(S[i, j]), i, j) for i, j in zip(iu[0][cross], iu[1][cross])),
                   key=lambda t: -t[0])
    dup = [t for t in pairs if t[0] >= 0.90 and t[1] >= 0.80]
    all_dup = [(float(C[i, j]), float(S[i, j]), i, j) for i, j in zip(*iu) if C[i, j] >= 0.90 and S[i, j] >= 0.80]
    parent = list(range(len(allp)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for _, _, i, j in all_dup:
        parent[find(i)] = find(j)
    groups = {}
    for k in range(len(allp)):
        groups.setdefault(find(k), []).append(allp[k][1].name)
    groups = [sorted(g) for g in groups.values() if len(g) > 1]
    if out_dir is not None:
        import json
        (Path(out_dir) / "duplicate_groups.json").write_text(json.dumps(groups, indent=1))
        with (Path(out_dir) / "near_duplicates.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["split_a", "image_a", "split_b", "image_b", "image_similarity", "strip_similarity", "likely_duplicate"])
            for c, st_, i, j in pairs[:300]:
                wr.writerow([allp[i][0], allp[i][1].name, allp[j][0], allp[j][1].name, round(c, 4), round(st_, 4),
                             c >= 0.90 and st_ >= 0.80])
    near = [(allp[i][0], allp[i][1].name, allp[j][0], allp[j][1].name, c, st_) for c, st_, i, j in dup]
    L.append(f"- Image pairs across splits compared (contrast-equalised): {len(pairs)}; image similarity median "
             f"{np.median([t[0] for t in pairs]):.2f}, 99.9th percentile {np.percentile([t[0] for t in pairs], 99.9):.2f}")
    L.append(f"- Likely re-exports of the same radiograph across splits (image >= 0.90 and text strip >= 0.80): {len(near)}")
    for s1, a, s2, b, c, st_ in near[:25]:
        L.append(f"    - {s1}/{a} ~ {s2}/{b} (image {c:.3f}, strip {st_:.3f})")
    L.append(f"- Duplicate groups in the whole dataset (any split): {len(groups)} groups, "
             f"{sum(len(g) for g in groups)} images (saved to duplicate_groups.json)")
    if out_dir is not None and pairs:
        tiles = []
        for c, st_, i, j in pairs[:24]:
            ia = cv2.resize(cv2.imread(str(allp[i][1])), (480, 240))
            ib = cv2.resize(cv2.imread(str(allp[j][1])), (480, 240))
            bar = np.zeros((22, 964, 3), np.uint8)
            tag = "LIKELY DUPLICATE" if (c >= 0.90 and st_ >= 0.80) else "different?"
            cv2.putText(bar, f"{allp[i][0]}/{allp[i][1].name} vs {allp[j][0]}/{allp[j][1].name}  image {c:.3f}  "
                        f"strip {st_:.3f}  {tag}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(np.vstack([bar, np.hstack([ia, np.full((240, 4, 3), 255, np.uint8), ib])]))
        cv2.imwrite(str(Path(out_dir) / "near_duplicates.jpg"), np.vstack(tiles))
        L.append("- The 24 most similar cross-split pairs are in near_duplicates.jpg (all scores in near_duplicates.csv): "
                 "same scan = same burned-in date/time, teeth, fillings and gaps.")
    counts = ", ".join(f"{s} {len(v)}" for s, v in files.items())
    return ok and not near, L, counts, near


def audit(run):
    run = Path(run)
    a = yaml.safe_load((run / "args.yaml").read_text())
    lr = None
    res = run / "results.csv"
    if res.exists():
        rows = list(csv.DictReader(res.open()))
        col = [k for k in rows[0] if k.strip() == "lr/pg0"]
        if col:
            lr = max(float(r[col[0]]) for r in rows)
    return a, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs", nargs="*", default=[], help="run folders (with args.yaml)")
    ap.add_argument("--out", default="results/integrity.md")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ok, lines, counts, near = check_splits(args.data_root, Path(args.out).parent)
    L = ["## Split integrity", "", f"Split sizes: {counts}.", ""] + lines + ["",
         ("**No leakage found**" if ok else "**Possible leakage: see above**") +
         " at the file and image level. The Kaggle export carries no patient identifiers, so images of the same "
         "patient taken on different visits cannot be grouped and could not be checked; this is a stated "
         "limitation. Cross-source generalisation is untested: all images come from one dataset; a second "
         "source (e.g. the Tufts Dental Database, by access request) would be needed."]

    if args.runs:
        L += ["", "## Baseline configuration audit", "",
              "| Setting | " + " | ".join(Path(r).name for r in args.runs) + " |",
              "| --- | " + " | ".join("---" for _ in args.runs) + " |"]
        audits = [audit(r) for r in args.runs]
        keys = ["model", "imgsz", "epochs", "batch", "optimizer", "lr0", "seed", "deterministic", "iou", "conf",
                "split", "fliplr", "mosaic", "close_mosaic", "hsv_v", "scale", "amp", "data"]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(str(a.get(k)) for a, _ in audits) + " |")
        L.append("| peak learning rate used (results.csv) | " + " | ".join(
            f"{lr:.6f}" if lr else "n/a" for _, lr in audits) + " |")
        splits = {str(a.get("split")) for a, _ in audits}
        L += ["", f"- Checkpoint selection split: {', '.join(sorted(splits))}"
              + (" (validation only: the test split never chose a checkpoint)." if splits == {"val"} else
                 " **WARNING: not the validation split.**"),
              "- Input: long side resized to imgsz, aspect kept, padded with gray "
              f"{PAD_VALUE}; pixels scaled to 0-1, BGR->RGB (Ultralytics; predict.py uses the same, see tests).",
              "- Anchors: none. YOLO11 heads are anchor-free, so there are no anchor settings to tune.",
              "- NMS IoU during validation = the iou row above (0.7 default); confidence 0.001 for mAP.",
              "- optimizer 'auto' picks AdamW for short runs and ignores lr0; the peak rate actually used is shown."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
