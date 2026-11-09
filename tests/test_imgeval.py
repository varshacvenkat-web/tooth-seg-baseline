"""Per-tooth matching and failure categories (src/imgeval.py) on hand-built cases."""
import numpy as np

from src.imgeval import counts, evaluate_frame, image_score

S = 200


def box(x, y, w=20, h=60):
    m = np.zeros((S, S), bool)
    m[y:y + h, x:x + w] = True
    return m


def run(gt, gn, pm, pn, pc=None):
    pc = pc if pc is not None else np.linspace(0.9, 0.5, len(pm)) if len(pm) else np.zeros(0)
    return evaluate_frame(np.stack(gt), gn, np.stack(pm) if pm else np.zeros((0, S, S), bool), pc, pn)


def test_perfect_prediction():
    g, p = run([box(10, 10), box(40, 10)], [8, 9], [box(10, 10), box(40, 10)], [8, 9])
    assert all(r["ok"] for r in g) and all(r["kind"] == "tp" for r in p)
    assert image_score(g) == 1.0


def test_missed_tooth():
    g, p = run([box(10, 10), box(100, 10)], [8, 9], [box(10, 10)], [8])
    assert [r["category"] for r in g] == ["ok", "missed"]
    assert image_score(g) == 0.5


def test_mirror_wrong_number():
    g, p = run([box(10, 10)], [3], [box(10, 10)], [14])         # tooth 3 called 14 (its mirror)
    assert g[0]["category"] == "wrong number (mirror)" and p[0]["kind"] == "wrong_number"


def test_neighbour_wrong_number():
    g, _ = run([box(10, 10)], [7], [box(10, 10)], [6])
    assert g[0]["category"] == "wrong number (neighbour)"
    g, _ = run([box(10, 10)], [8], [box(10, 10)], [9])          # central incisors: adjacent AND mirrors
    assert g[0]["category"] == "wrong number (neighbour)"


def test_duplicate_and_phantom_predictions():
    g, p = run([box(10, 10)], [8], [box(10, 10), box(11, 10), box(150, 120)], [8, 8, 30])
    assert [r["kind"] for r in p] == ["tp", "duplicate", "phantom"]
    c, fp, fn = counts([{"gt": g, "pred": p}])
    assert c["tp"] == 1 and c["fp"] == 2 and c["fn"] == 0 and fp == {"duplicate": 1, "phantom": 1}


def test_merged_neighbours():
    g, _ = run([box(10, 10), box(30, 10)], [8, 9], [box(10, 10, w=40)], [8])   # one mask over both
    assert g[1]["category"] == "merged neighbours"


def test_poor_boundary():
    g, _ = run([box(10, 10, h=80)], [8], [box(10, 10, h=30)], [8])            # IoU 0.375, right tooth
    assert g[0]["category"] == "poor boundary" and 0.3 < g[0]["iou"] < 0.5
