#!/usr/bin/env bash
#
# run_config.sh — run ONE Experiment A configuration end-to-end and record it.
#
# Runs ON THE JETSON (calls nvpmodel / jetson_clocks / trtexec — Jetson-only tools).
# Does, in order: set+lock power mode -> cooldown -> log start temp -> start tegrastats
# -> run trtexec -> stop tegrastats -> parse both logs -> append one CSV row.
#
# Usage:
#   ./run_config.sh <model> <precision> <mode> <run#>
#     model      resnet18 | resnet50 | vgg16 | ...   (expects ~/models/<model>.onnx)
#     precision  fp32 | fp16 | int8
#     mode       7w | 15w | 25w | maxn
#     run#       1 | 2 | 3   (the R-repeat index)
#
# Example:
#   ./run_config.sh resnet50 fp16 maxn 1
#
# Config (override via env): MODELS_DIR, LOG_DIR, CSV, COOLDOWN, ITER, AVGRUNS, WARMUP
set -euo pipefail

# ---- args --------------------------------------------------------------------
if [ "$#" -ne 4 ]; then
    echo "usage: $0 <model> <precision:fp32|fp16|int8> <mode:7w|15w|25w|maxn> <run#>" >&2
    exit 1
fi
MODEL="$1"; PRECISION="$2"; MODE="$3"; RUN="$4"

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
LOG_DIR="${LOG_DIR:-$HOME/logs/expA}"
CSV="${CSV:-$HOME/results_expA.csv}"
COOLDOWN="${COOLDOWN:-60}"      # seconds
ITER="${ITER:-1000}"
AVGRUNS="${AVGRUNS:-100}"
WARMUP="${WARMUP:-2000}"        # ms
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
PARSER="${PARSER:-$HOME/parse_tegrastats.py}"

# ---- map friendly names to flags / IDs ---------------------------------------
case "$PRECISION" in
    fp32) PREC_FLAG="" ;;
    fp16) PREC_FLAG="--fp16" ;;
    int8) PREC_FLAG="--int8" ;;
    *) echo "bad precision: $PRECISION (fp32|fp16|int8)" >&2; exit 1 ;;
esac

# nvpmodel IDs CONFIRMED on this unit (JP 6.2.1): 0=15W 1=25W 2=MAXN_SUPER 3=7W
case "$MODE" in
    15w)  NVP_ID=0 ;;
    25w)  NVP_ID=1 ;;
    maxn) NVP_ID=2 ;;
    7w)   NVP_ID=3 ;;
    *) echo "bad mode: $MODE (7w|15w|25w|maxn)" >&2; exit 1 ;;
esac

ONNX="$MODELS_DIR/$MODEL.onnx"
[ -f "$ONNX" ] || { echo "model not found: $ONNX" >&2; exit 1; }

RUN_ID="${MODEL}_${PRECISION}_${MODE}_run${RUN}"
mkdir -p "$LOG_DIR"
TRT_LOG="$LOG_DIR/${RUN_ID}.trtexec.log"
TEGRA_LOG="$LOG_DIR/${RUN_ID}.tegrastats.log"

echo "=== $RUN_ID ==="

# ---- 1. set + lock the operating point ---------------------------------------
sudo nvpmodel -m "$NVP_ID" >/dev/null
sudo jetson_clocks
echo "power mode set:"; sudo nvpmodel -q | tail -n +1

# ---- 2. cooldown + start temperature -----------------------------------------
echo "cooldown ${COOLDOWN}s..."
sleep "$COOLDOWN"
# Some Orin thermal zones intermittently return EAGAIN ("Resource temporarily
# unavailable"); tolerate that and take the hottest zone that DID read.
start_temp_mc=$( { cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null || true; } | sort -n | tail -1 )
start_temp_mc=${start_temp_mc:-0}
start_temp=$(awk "BEGIN{printf \"%.1f\", $start_temp_mc/1000}")   # milli-C -> C
echo "start temp: ${start_temp} C"

# ---- 3. tegrastats logging in the background ---------------------------------
sudo tegrastats --interval 100 --logfile "$TEGRA_LOG" &
TEGRA_PID=$!
sleep 0.3   # let it start logging before the workload

# ---- 4. the benchmark --------------------------------------------------------
# shellcheck disable=SC2086  # PREC_FLAG is intentionally word-split (may be empty)
"$TRTEXEC" --onnx="$ONNX" $PREC_FLAG \
    --iterations="$ITER" --avgRuns="$AVGRUNS" --warmUp="$WARMUP" \
    2>&1 | tee "$TRT_LOG"

# ---- 5. stop logging ---------------------------------------------------------
sudo kill "$TEGRA_PID" 2>/dev/null || true
wait "$TEGRA_PID" 2>/dev/null || true

# ---- 6. extract metrics ------------------------------------------------------
set +e   # a missing field should leave a blank cell, not kill the run
# trtexec: use GPU Compute Time (pure kernel), not the "Latency" line
gpu_line=$(grep "GPU Compute Time:" "$TRT_LOG" | tail -1)
med=$(grep -oP 'median = \K[0-9.]+'          <<< "$gpu_line")
p95=$(grep -oP 'percentile\(95%\) = \K[0-9.]+' <<< "$gpu_line")
p99=$(grep -oP 'percentile\(99%\) = \K[0-9.]+' <<< "$gpu_line")
thr=$(grep -oP 'Throughput: \K[0-9.]+'   "$TRT_LOG" | tail -1)
build=$(grep -oP 'Engine built in \K[0-9.]+' "$TRT_LOG" | tail -1)

# tegrastats parser: mean/peak power, peak RAM, peak GPU%
parsed=$(python3 "$PARSER" "$TEGRA_LOG")
mean_p=$(grep -oP 'mean_power_W=\K[0-9.]+'  <<< "$parsed")
peak_p=$(grep -oP 'peak_power_W=\K[0-9.]+'  <<< "$parsed")
peak_ram=$(grep -oP 'peak_ram_MB=\K[0-9]+'  <<< "$parsed")
gpu_util=$(grep -oP 'peak_gpu_pct=\K[0-9]+' <<< "$parsed")

# peak junction temp over the run
max_temp=$(grep -oE 'tj@[0-9.]+C' "$TEGRA_LOG" | grep -oE '[0-9.]+' | sort -n | tail -1)

# energy per inference (mJ) = mean power (W) * median latency (ms)
energy=$(awk "BEGIN{printf \"%.2f\", $mean_p * $med}")

# throttle flag: >80 C threshold (§5.2)
throttled=$(awk "BEGIN{print ($max_temp > 80) ? \"yes\" : \"no\"}")

# ---- 7. append one CSV row ---------------------------------------------------
# header matches data/results_expA.csv
if [ ! -f "$CSV" ]; then
    echo "run_id,datetime,model,params_M,precision,power_mode,jetson_clocks,start_temp_C,max_temp_C,throttled,median_L_ms,p95_L_ms,p99_L_ms,throughput_fps,mean_power_W,peak_power_W,energy_mJ,peak_ram_MB,gpu_util_pct,engine_build_s,trtexec_log,tegrastats_log,notes" > "$CSV"
fi
now=$(date -Iseconds) 
echo "$RUN_ID,$now,$MODEL,,$PRECISION,$MODE,yes,$start_temp,$max_temp,$throttled,$med,$p95,$p99,$thr,$mean_p,$peak_p,$energy,$peak_ram,$gpu_util,$build,$(basename "$TRT_LOG"),$(basename "$TEGRA_LOG")," >> "$CSV"

echo "--- recorded ---"
echo "median=${med}ms p95=${p95} p99=${p99} throughput=${thr}fps"
echo "power: mean=${mean_p}W peak=${peak_p}W  energy=${energy}mJ  ram=${peak_ram}MB  gpu=${gpu_util}%  maxT=${max_temp}C  throttled=${throttled}  build=${build}s"
echo "logs: $TRT_LOG , $TEGRA_LOG"
echo "csv:  $CSV"
