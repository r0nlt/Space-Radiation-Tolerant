#!/bin/bash
set -euo pipefail

# Long-run GA+QD experiment runner
# - Runs the AutoArch example with QD/MAP-Elites
# - Saves stdout to a timestamped log
# - Moves CSV outputs to results/genetic_algorithm/long_runs/

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULT_DIR="$ROOT_DIR/results/genetic_algorithm/long_runs"
mkdir -p "$RESULT_DIR"

# Tunables (override via env if desired)
TRIALS=${TRIALS:-50}
GENS=${GENS:-20}
POP=${POP:-20}
WIDTHS=${WIDTHS:-32,64,128,256,512}
SAVE_GEN=${SAVE_GEN:-1}
SAVE_ITER=${SAVE_ITER:-5}

TS=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$RESULT_DIR/ga_qd_long_${TS}.log"

echo "[INFO] Running long GA+QD experiment: trials=$TRIALS gens=$GENS pop=$POP widths=$WIDTHS" | tee "$LOG_FILE"

# Resolve executable location (allow override via EXE)
EXE=${EXE:-"$ROOT_DIR/examples/auto_arch_search_example"}
if [ ! -x "$EXE" ]; then
  ALT_EXE="$ROOT_DIR/space-radiation-tolerant/examples/auto_arch_search_example"
  if [ -f "$ALT_EXE" ]; then
    chmod +x "$ALT_EXE" || true
    EXE="$ALT_EXE"
  fi
fi
if [ ! -x "$EXE" ]; then
  echo "[ERROR] example binary not found: $EXE (also tried space-radiation-tolerant). Set EXE=/path/to/auto_arch_search_example" | tee -a "$LOG_FILE"
  exit 1
fi

pushd "$ROOT_DIR" >/dev/null
"$EXE" \
  --qd --adv-qd \
  --trials "$TRIALS" \
  --gens "$GENS" \
  --pop "$POP" \
  --save-gen "$SAVE_GEN" \
  --save-iter "$SAVE_ITER" \
  --widths "$WIDTHS" 2>&1 | tee -a "$LOG_FILE"

# Collect outputs if present
if [ -f "auto_arch_search_results.csv" ]; then
  cp auto_arch_search_results.csv "$RESULT_DIR/auto_arch_search_results_${TS}.csv"
fi
if [ -f "run_summaries.csv" ]; then
  cp run_summaries.csv "$RESULT_DIR/run_summaries_${TS}.csv"
fi
if [ -f "operator_stats.csv" ]; then
  cp operator_stats.csv "$RESULT_DIR/operator_stats_${TS}.csv"
fi

echo "[INFO] Long-run outputs saved under $RESULT_DIR" | tee -a "$LOG_FILE"
popd >/dev/null
