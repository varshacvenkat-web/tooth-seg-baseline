"""Stress test (no retraining): evaluate the baseline on degraded copies of the TEST images.

Degradations are applied at the model's input scale (long side 640), so their strength is the
same for every X-ray whatever its original size. A "clean" copy goes through the same resize
and PNG save without any degradation, so every drop is measured against it.
  blur      Gaussian blur, sigma 1.5 and 3 px
  noise     additive Gaussian noise, sigma 10 and 25 grey levels (fixed random seed)
  contrast  contrast scaled to 60% and 30% around each image's mean

Example:
    python src/degrade.py --data data/teeth_yolo/data.yaml --data-root data/teeth_yolo \
        --runs runs/baseline_s0/weights/best.pt runs/baseline_s1/weights/best.pt --out results/robustness
"""
import argparse
import shutil
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analyze import locked  # noqa: E402
from src.common import IMGSZ, VAL_CONF, load_image  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.imgeval import split_images  # noqa: E402

VARIANTS = {
    "clean": lambda im, rng: im,
    "blur sigma 1.5": lambda im, rng: cv2.GaussianBlur(im, (0, 0), 1.5),
    "blur sigma 3": lambda im, rng: cv2.GaussianBlur(im, (0, 0), 3),
    "noise sigma 10": lambda im, rng: np.clip(im + rng.normal(0, 10, im.shape[:2])[..., None], 0, 255).astype(np.uint8),
    "noise sigma 25": lambda im, rng: np.clip(im + rng.normal(0, 25, im.shape[:2])[..., None], 0, 255).astype(np.uint8),
    "contrast 60%": lambda im, rng: np.clip(im.mean() + 0.6 * (im.astype(np.float32) - im.mean()), 0, 255).astype(np.uint8),
    "contrast 30%": lambda im, rng: np.clip(im.mean() + 0.3 * (im.astype(np.float32) - im.mean()), 0, 255).astype(np.uint8),
}


def slug(name):
    return name.replace(" ", "_").replace("%", "pct").replace(".", "p")


def build(data_yaml, data_root, work):
    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d.get("path", data_root)).resolve()
    imgs = split_images(data_root, "test")
    yamls = {}
    for name, fn in VARIANTS.items():
        vdir = Path(work) / slug(name)
        (vdir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (vdir / "labels" / "test").mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for p in imgs:
            im = load_image(p)
            h, w = im.shape[:2]
            r = IMGSZ / max(h, w)
            if r != 1:
                im = cv2.resize(im, (round(w * r), round(h * r)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(vdir / "images" / "test" / f"{p.stem}.png"), fn(im, rng))
            shutil.copy(Path(data_root) / "labels" / "test" / f"{p.stem}.txt", vdir / "labels" / "test" / f"{p.stem}.txt")
        y = dict(d, path=str(vdir.resolve()), train=str(root / d["train"]), val=str(root / d["val"]), test="images/test")
        yamls[name] = vdir / "data.yaml"
        yamls[name].write_text(yaml.safe_dump(y, sort_keys=False))
    return yamls, imgs[0].stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--work", default="data/degraded")
    ap.add_argument("--out", default="results/robustness")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, iou = locked()
    yamls, example = build(args.data, args.data_root, args.work)

    res = {name: [] for name in VARIANTS}
    for w in args.runs:
        for name, y in yamls.items():
            m = evaluate(w, str(y), "test", VAL_CONF, iou)
            res[name].append((float(m.seg.map50), float(m.seg.map)))
            print(name, w, res[name][-1])
    sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0  # noqa: E731
    L = [f"## Robustness: baseline on degraded test images (mean ± SD over {len(args.runs)} seeds)", "",
         "| Test set | Mask mAP@50 | Mask mAP@50:95 | Drop in mAP@50:95 vs clean |", "| --- | --- | --- | --- |"]
    clean = [x[1] for x in res["clean"]]
    for name, v in res.items():
        a, b = [x[0] for x in v], [x[1] for x in v]
        drop = [c - x for c, x in zip(clean, b)]
        L.append(f"| {name} | {st.mean(a):.3f} ± {sd(a):.3f} | {st.mean(b):.3f} ± {sd(b):.3f} | "
                 + ("—" if name == "clean" else f"{st.mean(drop):.3f} ± {sd(drop):.3f} ({100 * st.mean(drop) / st.mean(clean):.0f}%)") + " |")
    L += ["", "Degradations applied after resizing to the 640 px model input; 'clean' = same resize, no degradation."]
    (out / "robustness.md").write_text("\n".join(L) + "\n")
    tiles = []
    for name, y in yamls.items():
        im = cv2.imread(str(Path(y).parent / "images" / "test" / f"{example}.png"))
        im = cv2.resize(im, (480, int(im.shape[0] * 480 / im.shape[1])))
        cv2.rectangle(im, (0, 0), (im.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(im, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(im)
    tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(out / "degradation_examples.jpg"),
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)]))
    print("\n".join(L))


if __name__ == "__main__":
    main()
