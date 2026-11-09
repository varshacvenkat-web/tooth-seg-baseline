# Experiment 1: how to pre-register

1. Run `bash reproduce.sh data baseline analyze` and read `results/failures/failures.md`
   (failure counts for the 20 worst test images and the estimated mAP impact).
2. Pick ONE change aimed at the top failure type. Starting points:

| Top failure type | One change to test | `change:` in exp1.yaml | Training cost |
| --- | --- | --- | --- |
| Poor boundary, missed small teeth / root tips | Higher input resolution | `{imgsz: 1024}` | ~2.5x per run |
| Wrong number (mirror: left/right swaps) | No horizontal flipping (tooth numbers encode side) | `{fliplr: 0.0}` | same |
| Metallic / restoration | Stronger brightness jitter | `{hsv_v: 0.6}` | same |
| Merged neighbours | Higher input resolution (more pixels between crowns) | `{imgsz: 1024}` | ~2.5x per run |
| Edge clipping | Less aggressive scale augmentation | `{scale: 0.25}` | same |

3. Fill in `experiments/exp1.yaml`: hypothesis, the single change, the metric and failure count
   you expect to move, and what result rejects the hypothesis. Example:

```yaml
name: exp1
hypothesis: "Most failures in the worst 20 are poor boundaries; a 1024 px input keeps boundary detail lost at 640."
change: {imgsz: 1024}
expected: "Mask mAP@50:95 rises by more than the baseline seed SD; poor-boundary teeth in the worst 20 fall."
reject_if: "Mean mAP@50:95 gain is within one baseline seed SD, or the paired comparison shows no consistent gain across seeds."
seeds: [0, 1, 2]
signed_off: false
```

4. Share it with the team. After sign-off set `signed_off: true`, commit it, then run
   `bash reproduce.sh exp1`. Same split (splits.json), same seeds, same evaluation; only the one
   setting differs. An image-size change is evaluated at the size it was trained at.
5. Report the outcome from `results/comparison.md`, including a null or negative result.
