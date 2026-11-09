#!/usr/bin/env bash
# One command reproduces everything:   bash reproduce.sh
# Or run stages:                        bash reproduce.sh data baseline analyze demo tests
#                                       bash reproduce.sh exp1        (after experiments/exp1.yaml is signed off)
# Quick CPU check on fake data:         SMOKE=1 bash reproduce.sh
#
# Settings (environment variables):
#   SEEDS="0 1 2"      seeds for baseline and experiment (same seeds, paired)
#   RUNS_DIR=runs      where trained runs live (point at Google Drive in Colab to keep them)
#   DATA_DIR=data      dataset folder
#   DEMO_IMAGES=       optional folder of new X-rays for the predict.py demo
#   KAGGLE_API_TOKEN   needed only if the dataset has not been downloaded yet
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=${SEEDS:-"0 1 2"}
RUNS_DIR=${RUNS_DIR:-runs}
DATA_DIR=${DATA_DIR:-data}
RAW="$DATA_DIR/Teeth Segmentation JSON"
YOLO_DIR="$DATA_DIR/teeth_yolo"
DATA="$YOLO_DIR/data.yaml"
RES=results
export RUNS_DIR

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo ">> SMOKE mode: fake data, tiny training. Numbers are meaningless."
  DATA_DIR=smoke_data; RAW="$DATA_DIR/fake_kaggle"; YOLO_DIR="$DATA_DIR/teeth_yolo"; DATA="$YOLO_DIR/data.yaml"
  RES=smoke_results; RUNS_DIR=smoke_runs; export RUNS_DIR
  export EXTRA_SET="epochs=${SMOKE_EPOCHS:-3} imgsz=320 batch=8 workers=0 plots=False"
  export POSTPROCESS_YAML=smoke_results/postprocess.yaml
  export BASE_MODEL=yolo11n-seg.pt
  SEEDS=${SEEDS_SMOKE:-"0 1 2"}
fi

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(data baseline analyze demo tests exp1)

weights() { for s in $SEEDS; do echo "$RUNS_DIR/$1_s$s/weights/best.pt"; done; }
group_runs() { for s in $SEEDS; do echo "$2=$RUNS_DIR/$1_s$s/weights/best.pt"; done; }

stage_data() {
  if [[ "${SMOKE:-0}" == "1" ]]; then
    [[ -d "$RAW" ]] || python scripts/make_fake_kaggle.py "$RAW"
  elif [[ ! -d "$RAW" ]]; then
    : "${KAGGLE_API_TOKEN:?Dataset missing and KAGGLE_API_TOKEN not set}"
    kaggle datasets download -d humansintheloop/teeth-segmentation-on-dental-x-ray-images -p "$DATA_DIR" --unzip -q
  fi
  if [[ ! -f "$DATA" ]]; then
    if [[ -f splits.json && "${SMOKE:-0}" != "1" ]]; then
      python src/convert.py --src "$RAW" --out "$YOLO_DIR" --split-file splits.json   # the committed split
    else
      python src/convert.py --src "$RAW" --out "$YOLO_DIR" --seed 42
    fi
  fi
  if [[ "${SMOKE:-0}" != "1" && ! -f splits.json ]]; then
    cp "$YOLO_DIR/splits.json" splits.json   # commit this file so everyone uses the identical split
  fi
  python src/check_integrity.py --data-root "$YOLO_DIR" --out "$RES/integrity.md" || true
}

stage_baseline() {
  for s in $SEEDS; do
    python src/train.py --data "$DATA" --seed "$s" --name "baseline_s$s"
  done
  python src/evaluate.py --data "$DATA" --split test --out "$RES/baseline_test.csv" --runs $(group_runs baseline baseline)
  python src/sweep.py --data "$DATA" --data-root "$YOLO_DIR" --out "$RES/sweep" --runs $(weights baseline)
  IOU=$(python -c "import os,yaml;print(yaml.safe_load(open(os.environ.get('POSTPROCESS_YAML','configs/postprocess.yaml')))['iou'])")
  python src/evaluate.py --data "$DATA" --split test --iou "$IOU" --out "$RES/tuned_test.csv" \
    --runs $(group_runs baseline "tuned")
}

stage_analyze() {
  python src/analyze.py --data "$DATA" --data-root "$YOLO_DIR" --out "$RES/analysis" --runs $(weights baseline)
  python src/failures.py --data "$DATA" --data-root "$YOLO_DIR" --out "$RES/failures" --runs $(weights baseline)
  python src/degrade.py --data "$DATA" --data-root "$YOLO_DIR" --work "$DATA_DIR/degraded" \
    --out "$RES/robustness" --runs $(weights baseline)
  python src/check_integrity.py --data-root "$YOLO_DIR" --out "$RES/integrity.md" \
    --runs $(for s in $SEEDS; do echo "$RUNS_DIR/baseline_s$s"; done) || true
}

stage_demo() {
  first=$(weights baseline | head -1)
  python src/demo.py --weights "$first" --data-root "$YOLO_DIR" --out "$RES/demo" \
    --failures-csv "$RES/failures/failures_worst20.csv" ${DEMO_IMAGES:+--images "$DEMO_IMAGES"}
}

stage_tests() {
  python -m pytest -q tests | tee "$RES/pytest.txt"
}

stage_exp1() {
  if [[ "${SMOKE:-0}" == "1" ]]; then CHANGE="fliplr=0.0"; else CHANGE=$(python - <<'EOF'
import sys, yaml
e = yaml.safe_load(open("experiments/exp1.yaml"))
if not e.get("signed_off"):
    sys.exit("experiments/exp1.yaml is not signed off yet: fill it in, get team sign-off, set signed_off: true")
ch = e.get("change") or {}
if len(ch) != 1:
    sys.exit(f"exp1 must change exactly one setting, got {ch}")
print(" ".join(f"{k}={v}" for k, v in ch.items()))
EOF
); fi
  for s in $SEEDS; do
    python src/train.py --data "$DATA" --seed "$s" --name "exp1_s$s" --set $CHANGE
  done
  python src/evaluate.py --data "$DATA" --split test --out "$RES/exp1_test.csv" --runs $(group_runs exp1 exp1)
  python src/compare.py --baseline "$RES/baseline_test.csv" --out "$RES/comparison.md" \
    --other "Tuned post-processing=$RES/tuned_test.csv" "Exp 1 ($CHANGE)=$RES/exp1_test.csv"
}

mkdir -p "$RES"
{ python --version; pip freeze 2>/dev/null; python -c "import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"; } > "$RES/environment.txt" 2>&1 || true
for st in "${STAGES[@]}"; do
  echo; echo "================ stage: $st ================"
  "stage_$st"
done
echo; echo "Done. Results in $RES/"
