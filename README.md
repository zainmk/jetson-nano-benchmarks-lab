# Jetson Orin Nano Benchmarks Lab

Benchmarking neural-network inference on the **NVIDIA Jetson Orin Nano Super Developer Kit** — how model architecture, numerical precision (FP32 / FP16 / INT8), and power mode (7 W / 15 W / MAXN "Super") trade off against latency, throughput, and energy efficiency on edge hardware.

**Environment:** JetPack 6.2.1 · L4T 36.4.7 · CUDA 12.6 · TensorRT 10.3 · Ampere GPU (SM 8.7)

### Key findings — Experiment A (compute microbenchmark)
- **INT8 is 3.1–4.1× faster than FP32** (FP16 ~2.0×) on the Ampere Tensor Cores.
- **MAXN "Super" gives a 1.56× throughput uplift over 15 W** — short of NVIDIA's ~1.7× claim, because a batch-1 workload doesn't saturate the GPU.
- **MAXN is the *most* energy-efficient mode despite the highest power draw** ("race-to-idle" — finishing faster beats drawing less).
- No thermal throttling occurred (≤ 62 °C) with the stock cooler at 21 °C ambient.

![Precision speedup over FP32](images/fig3_precision_speedup.png)

### Status
- ✅ Experiment A — architecture × precision × power (Tables 1–2, Figures 1–4, discussion)
- ⏳ Experiment B — camera detection pipeline (Tables 4–5)
- ⏳ Table 3 — classification accuracy (labelled subset + INT8 calibration)

### Repo layout
- `lab.md` — full lab report (method, results, discussion, appendices)
- `data/` — raw per-run logs + parsed results CSVs
- `scripts/` — benchmark harness (`run_config.sh`)
- `images/` — figures
- `CLAUDE_CHAT_LOG.md` — how AI tooling was directed and verified during the build

`lab_template.md` is the blank template (built with AI assistance to balance learning and implementation); `lab.md` is the completed study.

### Equipment
- NVIDIA Jetson Orin Nano Super Developer Kit [🔗](https://www.amazon.ca/NVIDIA-Jetson-Orin-Nano-Developer/dp/B0BZJTQ5YP)
- USB camera (old / outdated) — for Experiment B
- 64 GB microSD card (Jetson OS boot media)
