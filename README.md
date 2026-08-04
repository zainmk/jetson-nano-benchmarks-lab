# Jetson Orin Nano Benchmarks Lab

### When the camera is the bottleneck

A measurement study of real-time object recognition on the **NVIDIA Jetson Orin Nano Super Developer Kit**, asking a deployment question rather than a benchmarking one: *once a cheap sensor — not the accelerator — bounds throughput, where does the remaining energy actually live?*

**Environment:** JetPack 6.2.1 · L4T 36.4.7 · CUDA 12.6 · TensorRT 10.3 · Ampere GPU (SM 8.7)



<img width="302" height="403" alt="IMG_3469" src="https://github.com/user-attachments/assets/ccd69b65-205e-4f72-b40a-e0054f61dc54" />



---

## The conclusion

**The Orin Nano Super is 5–8× over-provisioned for 30 fps detection from a USB webcam.** Once compute stops being the binding constraint, two standard pieces of edge-deployment advice **invert**.

**Power mode inverts.** When the GPU is saturated, MAXN "Super" is the *most* energy-efficient mode — finishing each inference faster converts into more work ("race-to-idle"). Under a fixed 30 fps camera, that reverses: at identical ~29 fps output, energy per frame falls monotonically as the mode is *reduced* — **212 mJ at MAXN → 184 at 15 W → 137 at 7 W**. MAXN burns **1.55× the energy for the same result**. Decomposing the power logs shows why: draw during GPU-active samples is indistinguishable from idle, so board power is essentially a **static function of the power mode**, not of the inference performed.

**Model selection inverts.** With every detector clearing 30 fps at 7 W, throughput stops discriminating and the choice falls to accuracy — which turns out to be effectively free. YOLOv8s recovers **2.3× as many objects** as SSD-Mobilenet-v2 while being **33% smaller** and costing negligible extra power. The recommended configuration is **YOLOv8s at 7 W**: the *largest* model that fits, in the *leanest* power mode — the opposite of the usual instinct to reach for the smallest network.

> **The general principle:** the correct configuration depends on which resource is scarce. Compute-bound, you pay per unit of *work*, and racing to idle wins. Sensor-bound, you pay per unit of *time*, and minimizing draw wins. A system characterized only in the first regime will be misconfigured in the second.

---

## Results at a glance

**Experiment A — compute microbenchmark** (33 runs, R = 3)
- **INT8 is 3.1–4.1× faster than FP32** (FP16 ~2.0×) on the Ampere Tensor Cores — gains specific to third-generation Tensor Cores that would not transfer to the original Maxwell Jetson Nano.
- **INT8 is also ~4× smaller**: VGG-16 drops from 528 MiB to 133 MiB.
- **MAXN "Super" delivered a measured 1.56× uplift over 15 W** against NVIDIA's claimed ~1.7× (≈92%) — the shortfall explained by a batch-1 workload that never saturates the power envelope.
- No thermal throttling anywhere (≤ 62 °C at 21 °C ambient with the stock cooler).

**Experiment B — camera pipeline**
- All three detectors (SSD-Mobilenet-v2, YOLOv8n, YOLOv8s) infer **5–8× faster than the camera delivers frames**; none is compute-bound.
- Capture rate is dominated by pixel format: raw YUYV at 720p exceeds USB 2.0 bandwidth (**9.6 fps**) while MJPG restores **~29 fps** — a 3× swing from the codec alone.
- Detection accuracy is monotonic in model capacity: **21% / 31% / 48%** class-presence recall for SSD / YOLOv8n / YOLOv8s.


<img width="1280" height="720" alt="yolo_frame_detection" src="https://github.com/user-attachments/assets/bc02a9d3-cd4c-4c88-a0c4-66f1c058cd34" />

![Precision speedup over FP32](images/fig3_precision_speedup.png)

---

## Measurement pitfalls documented

Part of the contribution is what it took to get numbers worth trusting. Each of these would have silently corrupted a result:

- **`detectnet` over-reports inference latency ~5×** (~21 ms reported vs ~4 ms measured) via a broken CUDA-event timer — the reason a custom timing harness exists.
- **Confidence thresholds differ by tool** (`detectNet` 0.5, Ultralytics 0.25); comparing detectors at their defaults credits one for detections the other would never emit.
- **`nvpmodel` power-mode IDs do not match vendor documentation** on this unit, and 7 W requires a reboot that hangs non-interactive automation on a hidden prompt.
- **Several Orin thermal zones intermittently return `EAGAIN`**, and `tegrastats`' GPU-utilization field reports peak rather than mean.

---

## Scope and honesty

The conclusions are **confirmatory, not novel** — that reduced clocks lower power draw, and that a slow sensor bottlenecks a fast accelerator, are established results. The contribution is quantitative characterization on this specific platform, and the methodology needed to obtain trustworthy numbers.

Known limitations are documented in `lab.md` §8, including: single device, class-presence rather than mAP scoring on a 10-frame clip, AI-assisted (author-verified) ground-truth labels, uneven repeat counts in the power sweep, and loose measurement windows in the pipeline power runs.

**Out of scope:** classification accuracy across precisions (Table 3) — it requires a labelled ImageNet subset and INT8 calibration, so H1/H2 are ruled on *speed only*.

---

## Repo layout

- `lab.md` — the full report (method, results, discussion, appendices)
- `data/` — raw per-run logs, results CSVs, and the accuracy clip ([`data/README.md`](data/README.md) explains the scheme)
- `scripts/` — the harness: `run_config.sh`, `bench_e2e.py`, `parse_tegrastats.py`, `score_accuracy.py`, `models_download.py`
- `images/` — figures
- `NOTES.md` — parked ideas and candidate future work
- `CLAUDE_CHAT_LOG.md` — how AI tooling was directed, verified, and corrected during the build

`lab_template.md` is the blank template (built with AI assistance to balance learning and implementation); `lab.md` is the completed study.

---

## Future work

The most informative next step is **locating the crossover** — scaling model complexity until a configuration can no longer hold 30 fps at each power mode, producing a minimum-viable-mode-versus-complexity curve that works as a design tool rather than a single operating point. Beyond that: isolating static from dynamic power, testing whether `jetson_clocks` (standard tuning advice) is counterproductive for sensor-capped deployment, and quantifying the peripheral/fan contribution to the static floor that dominates here.

---

## Equipment

- NVIDIA Jetson Orin Nano Super Developer Kit [🔗](https://www.amazon.ca/NVIDIA-Jetson-Orin-Nano-Developer/dp/B0BZJTQ5YP)
- USB webcam — generic, 1280×720 @ 30 fps (MJPG / YUYV)
- 64 GB microSD card (Jetson OS boot media)
