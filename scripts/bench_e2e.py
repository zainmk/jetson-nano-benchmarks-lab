#!/usr/bin/env python3
"""
bench_e2e.py -- Experiment B throughput harness (Table 4).

Measures three frame rates for a detection model on the live camera:
  capture FPS    -- raw camera frame-delivery rate (capture-only phase)
  infer FPS      -- pure GPU inference rate (per-frame wall-clock)
  end-to-end FPS -- full capture -> infer pipeline throughput

Uses time.perf_counter() (reliable) instead of detectNet's internal timer
(which emits the buggy 'device not ready (error 600)' on this container), and
cudaDeviceSynchronize() so asynchronous GPU work is fully accounted for.

RUN INSIDE THE CONTAINER (needs the jetson_inference module):
    python3 /jetson-inference/data/bench_e2e.py --model ssd-mobilenet-v2 --run 1

Applies to jetson-inference detectNet models: ssd-mobilenet-v2 (B1),
ssd-inception-v2 (B2). YOLOv8 (B3/B4) uses a separate ONNX/TensorRT path.

NOTE on precision: jetson-inference builds these SSD models at FP16 by default
(its 'fastest native precision'); INT8 for the .uff models would need a
calibration cache. We therefore record precision=fp16 for these runs.

Power/RAM/GPU-util (the remaining Table 4 columns) come from `tegrastats` run
on the HOST in a parallel SSH session during the run -- see the header comment
block below for the two-terminal procedure.
"""
import time, csv, argparse, os, statistics as st
from datetime import datetime
from jetson_inference import detectNet
from jetson_utils import videoSource, cudaDeviceSynchronize

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ssd-mobilenet-v2")
ap.add_argument("--input", default="/dev/video0")
ap.add_argument("--input-codec", default="mjpeg",
                help="camera codec: 'mjpeg' (compressed, fits USB bandwidth -> 30fps) "
                     "or 'raw' (uncompressed YUYV, bandwidth-limited). Set '' for file inputs.")
ap.add_argument("--input-width", type=int, default=1280)
ap.add_argument("--input-height", type=int, default=720)
ap.add_argument("--warmup", type=int, default=100, help="frames discarded before timing")
ap.add_argument("--frames", type=int, default=500, help="measured frames (>= ~500)")
ap.add_argument("--cap-frames", type=int, default=150, help="capture-only calibration frames")
ap.add_argument("--run", default="1", help="R-repeat index (1/2/3)")
ap.add_argument("--power-mode", default="maxn")
ap.add_argument("--precision", default="fp16", help="engine precision label for the record")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--csv", default="/jetson-inference/data/results_expB.csv")
args = ap.parse_args()

RUN_ID = f"{args.model}_{args.precision}_{args.power_mode}_run{args.run}"
print(f"=== {RUN_ID} ===")
print(f"loading {args.model} ...")
net = detectNet(args.model, threshold=args.threshold)
src_argv = [f"--input-width={args.input_width}", f"--input-height={args.input_height}"]
if args.input_codec:
    src_argv.append(f"--input-codec={args.input_codec}")
cam = videoSource(args.input, argv=src_argv)

# ---- phase 1: capture-only (true camera frame-delivery rate) -----------------
print("measuring capture rate ...")
n = 0
t_start = None
while n < args.cap_frames:
    img = cam.Capture()
    if img is None:
        continue
    if n == 0:
        t_start = time.perf_counter()
    n += 1
cap_elapsed = time.perf_counter() - t_start
capture_fps = round((args.cap_frames - 1) / cap_elapsed, 2)

# ---- phase 2: full pipeline (infer + end-to-end) -----------------------------
print(f"warming up {args.warmup} frames, then measuring {args.frames} ...")
inf_ms = []
seen = 0
measured = 0
e2e_start = None
while measured < args.frames:
    img = cam.Capture()
    if img is None:
        continue
    t1 = time.perf_counter()
    net.Detect(img, overlay="none")          # headless: no box drawing / encode
    cudaDeviceSynchronize()
    t2 = time.perf_counter()
    seen += 1
    if seen <= args.warmup:
        continue
    if measured == 0:
        e2e_start = t1                        # start the e2e clock at first measured frame
    inf_ms.append((t2 - t1) * 1000.0)
    measured += 1
e2e_elapsed = time.perf_counter() - e2e_start

infer_fps = round(1000.0 / st.mean(inf_ms), 2)
e2e_fps = round(args.frames / e2e_elapsed, 2)
infer_med_ms = round(st.median(inf_ms), 3)

# ---- record one CSV row (matches data/results_expB.csv header) ---------------
row = {
    "run_id": RUN_ID,
    "datetime": datetime.now().isoformat(timespec="seconds"),
    "model": args.model,
    "precision": args.precision,
    "power_mode": args.power_mode,
    "render": "no",
    "capture_fps": capture_fps,
    "infer_fps": infer_fps,
    "e2e_fps": e2e_fps,
    "gpu_util_pct": "",     # from tegrastats (host)
    "cpu_util_pct": "",     # from tegrastats (host)
    "peak_ram_mb": "",      # from tegrastats (host)
    "mean_power_W": "",     # from tegrastats (host)
    "correct": "", "missed": "", "false_pos": "", "mean_conf": "",  # Table 5 (accuracy)
    "clip_name": "", "notes": "",
}
header = list(row.keys())
new_file = not os.path.exists(args.csv)
with open(args.csv, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    if new_file:
        w.writeheader()
    w.writerow(row)

print("--- recorded ---")
print(f"capture={capture_fps} fps   infer-only={infer_fps} fps   end-to-end={e2e_fps} fps   "
      f"(infer median {infer_med_ms} ms)")
print(f"csv: {args.csv}")

# jetson_inference's Python bindings segfault during interpreter GC on exit
# (harmless -- data is already written). Exit immediately to skip that cleanup.
import sys
sys.stdout.flush()
os._exit(0)
