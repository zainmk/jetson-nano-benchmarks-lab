# Data — collection & tracking scheme

All measurement data for this study. The layout separates **immutable raw logs** from the
**parsed results** derived from them, so every figure in `lab.md` traces back to the log
that produced it.

## Layout

```
data/
├── README.md                   # this file
├── results_expA.csv            # Exp A compute microbenchmark      -> Tables 1, 2
├── results_expB_compute.csv    # Exp B detector compute sweep      -> Table 6
├── results_expB.csv            # Exp B camera pipeline             -> Tables 4, 7
├── clip_ground_truth.csv       # verified frame labels             -> Table 5
├── ssd_detections.log          # SSD output on the sampled frames  -> Table 5
├── run_metrics.txt             # parsed tegrastats values (pipeline runs)
├── clip/                       # the fixed accuracy clip (73 frames)
├── clip_sampled/               # the 10 scored frames
├── yolo_out/{n,s}/labels/      # YOLOv8 detections, one .txt per frame
├── env/                        # one-time environment snapshots
└── raw/                        # immutable per-run logs -- never edited after capture
    ├── expA/                   # 66 logs  (33 runs x trtexec + tegrastats)
    ├── expB_compute/           # 22 logs  (11 runs x trtexec + tegrastats)
    └── expB_pipeline/          # tegrastats logs for the Table 7 power sweep
```

## The three result CSVs — what distinguishes them

| File | Measures | Tool | Feeds |
|---|---|---|---|
| `results_expA.csv` | classifier compute: latency, throughput, power | `run_config.sh` → `trtexec` | Tables 1, 2 |
| `results_expB_compute.csv` | *detector* compute, no camera | `run_config.sh` → `trtexec` | Table 6 |
| `results_expB.csv` | camera pipeline: capture / infer / end-to-end fps | `bench_e2e.py` | Tables 4, 7 |

`results_expB_compute.csv` is named for measuring **compute only** within Experiment B —
YOLO cannot run through `bench_e2e.py` (jetson-inference's `detectNet` does not decode YOLO
output), so its infer-only rate was measured with the Experiment A harness on synthetic input.

## Naming convention

Run IDs are `{model}_{precision}_{mode}_run{N}`, e.g. `resnet50_int8_maxn_run2`. The
`trtexec` and `tegrastats` logs for one run share that stem (`.trtexec.log` /
`.tegrastats.log`), and the same ID is the `run_id` key in the CSV.

## Rigor (lab.md §5.2)

Warm-up discarded, N ≥ 1000 measured, **R = 3 repeats** per configuration, 60 s cooldown,
θ logged, `throttled` flagged above 80 °C. Clocks pinned with `jetson_clocks` for
reproducibility.

## Known gaps

- **Table 7 pipeline power is R = 1 at 15 W and 7 W** (R = 3 only at MAXN), and the
  `tegrastats` windows were started/stopped manually, so they include model loading and
  idle. Direction is robust; absolute values are approximate. See lab.md §8.
- **The SSD pipeline tegrastats logs were lost.** Their parsed values survive in
  `run_metrics.txt`, so the numbers are traceable but not re-auditable.
- **`env/` holds only `v4l2_formats.txt`.** `jetson_release` and `nvpmodel -p --verbose`
  output is recorded in lab.md Appendix C.3 rather than as separate files.

## Environment snapshots

```bash
jetson_release                             > env/jetson_release.txt
sudo nvpmodel -p --verbose                 > env/nvpmodel_p.txt
v4l2-ctl -d /dev/video0 --list-formats-ext > env/v4l2_formats.txt
```
