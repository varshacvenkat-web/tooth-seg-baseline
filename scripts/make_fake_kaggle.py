"""Make a small fake dataset in the Kaggle (Supervisely) layout, for a quick end-to-end smoke test
of reproduce.sh on a CPU (SMOKE=1). The numbers it produces mean nothing."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

NAMES = ['13', '14', '15', '11', '12', '19', '20', '21', '22', '23', '24', '25', '27', '32', '16', '26', '17',
         '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '18', '28', '29', '30', '31', '13(polygon)']


def main(out="data/fake_kaggle", n=40):
    out = Path(out)
    img_dir, mask_dir = out / "d2" / "img", out / "d2" / "masks_machine"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    (out / "obj_class_to_machine_color.json").write_text(json.dumps({n: [i + 1] * 3 for i, n in enumerate(NAMES)}))
    value = {int(n): NAMES.index(n) + 1 for n in NAMES if n.isdigit()}
    rng = np.random.default_rng(0)
    for k in range(n):
        h, w = 480, 960
        img = cv2.GaussianBlur(rng.integers(30, 80, (h, w)).astype(np.uint8), (0, 0), 2)
        m = np.zeros((h, w), np.uint8)
        for row, teeth in ((0, list(range(1, 17))), (1, list(range(32, 16, -1)))):
            for j, t in enumerate(teeth):
                if rng.random() < 0.1:
                    continue  # missing tooth
                x = 40 + j * 55 + int(rng.integers(-4, 5))
                y = 70 if row == 0 else 250
                axes = (20 + (6 if t in (1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32) else 0), 75)
                cv2.ellipse(img, (x + 22, y + 80), axes, 0, 0, 360, int(rng.integers(150, 200)), -1)
                cv2.ellipse(m, (x + 22, y + 80), axes, 0, 0, 360, value[t], -1)
                if rng.random() < 0.08:  # a bright "filling"
                    cv2.circle(img, (x + 22, y + 60 if row == 0 else y + 100), 12, 250, -1)
        cv2.imwrite(str(img_dir / f"{k}.jpg"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(mask_dir / f"{k}.png"), cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
    print(f"fake dataset: {n} images in {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
