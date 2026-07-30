# Working Log — Building the Jetson Benchmark Lab with an AI Assistant

*An edited excerpt of working sessions with Claude (Anthropic) while designing and running
this benchmarking study. It is included as a portfolio artifact to show **how** I use AI
tooling in a real technical workflow — not as a code generator, but as a reasoning partner
whose output I verify against the actual hardware and correct when it is wrong.*

*Condensed for readability and scrubbed of personal details. Every design decision,
correction, and dead end is faithful to the real sessions.*

---

## 1. Orienting: where does the project stand?

**Zain:** Read through the lab — what am I supposed to do next?

**Claude:** You're at the transition from **setup → data collection**. The device is confirmed
up (JetPack 6.2.1, CUDA 12.6, TensorRT 10.3); §6 Results is empty. The single most useful next
action is to confirm the power-mode IDs — nearly every experiment depends on which `nvpmodel -m`
maps to 7 W / 15 W / MAXN, and that mapping isn't guaranteed stable.

---

## 2. Verifying the power-mode mapping (the first correction)

**Zain:** *(pastes `nvpmodel -p` output)*

**Claude:** The template *assumed* MAXN = mode 2 and 15 W = mode 1. Your unit's real map:
0=15W, 1=25W (default), 2=MAXN_SUPER, 3=7W. Mode 2 was right; **15 W and 7 W are not modes
1/0** as the generic docs say. One nuance to note: 15 W runs the CPU *faster* than 25 W but caps
the GPU/memory much lower — so for GPU inference, 25 W ≫ 15 W despite the CPU number.

**Design decision:** Recorded the confirmed mapping in three places and corrected the
procedure's mode-selection commands, which had the wrong IDs. The whole H5 test depends on it.

---

## 3. Ambient temperature — measuring the right thing

**Zain:** What is the best way to collect the ambient temperature?

**Claude:** First, a correction: the ~41 °C value in the doc is the **SoC die temperature**, not
ambient. Ambient is the *room air* at the fan intake — a controlled variable *because* it's
independent of the chip. Use an external thermometer.

**Zain:** The number is 21.

**Design decision:** Caught a mislabelled confound — 21 °C ambient vs 41 °C die is the
difference between a meaningful control variable and a meaningless one.

---

## 4. Scoping the power-mode factor

**Zain:** Leave it as a three-level design — no major changes to the structure of the lab.

**Design decision:** Deliberately *declined* an available fourth power level (25 W) to keep the
design clean and matched to the hypothesis. More data points isn't automatically better.

---

## 5. What actually gets benchmarked — camera vs. dataset

**Zain:** Does the USB camera actually get used, or does inference run on a loaded dataset?

**Claude:** The camera is used in only one experiment. Experiment A's `trtexec` feeds **synthetic
random tensors** — the computation is identical to a real image, so it validly measures speed but
says nothing about accuracy. The camera only enters Experiment B (the deployment pipeline).

**Consequence:** Reordered the work — all of Experiment A can be collected with no camera, so I
started the compute benchmarks immediately and deferred camera setup to Experiment B.

---

## 6. A real dependency mismatch — the container tag

**Zain:** *(pastes)* `manifest for dustynv/jetson-inference:r36.4.7 not found`.

**Claude:** The launch script auto-generated a tag matching your exact L4T, but images are only
published for base releases. *(queries the tag list)* The images top out at **r36.3.0** — no
r36.4.x exists. Pin it explicitly; it runs on your r36.4.7 host via the NVIDIA container runtime.

**Design decision:** Rather than trust the auto-detected tag or downgrade blindly, verified the
published tag list, understood *why* the mismatch is safe, and recorded the pinned version.

---

## 7. The 7 W reboot trap (an hour-long "hang" debugged)

**Zain:** I automated the power sweep, but it's been stuck for over an hour on the first 7 W run.

**Claude:** An hour is a hang, not slow work — your slowest run all night was ~4 min. And notice
the output stopped *before* "power mode set" — so it's wedged on the very first step, switching to
7 W.

**Zain:** *(runs it manually)* `time sudo nvpmodel -m 3` → `Reboot required for changing to this
power mode: 3 … enter YES/yes to confirm:`

**Claude:** There it is. **7 W requires a reboot** (mode 3 takes CPU cores offline), so `nvpmodel`
prompts interactively. In the batch script that call was `… >/dev/null`, so the prompt was
*swallowed* — the script sat forever waiting on a `YES` you couldn't see. 15 W / 25 W / MAXN
switch live; only 7 W needs the reboot.

**Design decision:** Ran the 7 W runs **manually** (reboot into mode 3 first), documented it as a
platform limitation in §8 + the appendix, and **hardened the driver** to (a) skip the switch when
already in the target mode and (b) refuse to auto-switch *into* 7 W rather than hang. A real,
non-obvious hardware gotcha that anyone reproducing the study would otherwise hit blind.

---

## 8. Reading the results honestly

**Zain:** *(pastes the completed 33-run dataset)* How is it?

**Claude:** Clean and tight (R=3 repeats within ~1%, nothing throttled). The headline results:
FP16 ≈ 2.0× over FP32, INT8 a *further* 1.6–2.1×; the 15 W→MAXN "Super" uplift measured **1.56×**
vs NVIDIA's claimed ~1.7×.

**Zain:** Write the discussion — but keep the original research questions, and prompt me so I
understand each verdict.

**Claude:** Two things I flagged rather than glossed:
- **H1/H2 are supported on *speed only*.** I've shown INT8 is 3–4× faster, but *not* that it's
  still accurate — that needs the labelled-accuracy runs (Table 3), so the verdict is explicitly
  partial until then.
- **H5 undershot the 1.7× claim, and I explained *why*:** the ResNet-50 INT8 workload draws only
  ~8.8 W — nowhere near MAXN's 25 W envelope — so it doesn't saturate the GPU; the headline
  figure is measured on GPU-saturating workloads.

And one counterintuitive finding worth stating: **MAXN is the *most* energy-efficient mode despite
the highest power** — finishing each inference faster ("race-to-idle") beats drawing less.

**Design decision:** Marked accuracy (H1) and the deployment hypothesis (H4) as **pending** rather
than overclaim, and required every verdict to carry a number *and* a mechanism. Honest scoping is
what makes a discussion credible.

---

## 9. Experiment B — not trusting the tool's own timer

**Zain:** *(runs the camera detector)* It works — but why is the inference so slow?

**Claude:** The `detectnet` demo reports ~21 ms/frame, but it's spamming
`cudaEventElapsedTime … device not ready (error 600)` — its internal profiler is failing (a
CUDA-event bug, likely from the r36.3-container / r36.4.7-host gap). Don't trust that number.

**Zain:** So how do we measure it properly?

**Claude:** Instrument it ourselves — a small harness that times `net.Detect()` with
`perf_counter` + `cudaDeviceSynchronize`. *(built `bench_e2e.py`)* Result: **inference is ~4 ms
(≈240 fps), not 21 ms** — the demo's timer was over-reporting **~5×**.

**Design decision:** Built a custom timing harness rather than trust the tool's self-reported FPS,
and documented the discrepancy. The whole point of Experiment B is trustworthy numbers; a tool
that randomly reports zero can't provide them.

---

## 10. The camera-bandwidth bottleneck (H4, sharpened)

**Zain:** *(pastes harness output)* `capture=9.64 fps  infer-only=239 fps  end-to-end=9.66 fps`

**Claude:** Two revelations. Inference is 239 fps — fast. But **capture is only 9.6 fps**, and
that's the real bottleneck. The log shows jetson-inference selected **raw YUYV** — at 1280×720
that's ~55 MB/s, *above USB 2.0's ~35 MB/s*, so the camera can only push ~10 fps of raw video.
Switching to **MJPG** (compressed) fits the bandwidth and restores ~30 fps.

**Design decision:** Used MJPG for the realistic capture rate, but kept the YUYV result too — the
raw-vs-compressed difference (~3× on capture) is itself an H4 finding: for a lightweight detector
the bottleneck is **camera bandwidth, not compute** (~10–30 fps capture vs ~239 fps inference).
Also handled a benign segfault in the jetson-inference Python bindings (harmless on-exit GC crash)
with an explicit `os._exit(0)` so runs end cleanly.

---

## Reflection

The through-line across both experiments: the AI is fast at interpreting raw tool output,
recalling command syntax, and proposing structure — but a striking number of its initial
assumptions, and even the *tools' own self-reported measurements*, were **wrong for this specific
setup**: the power-mode IDs, the ambient-temperature source, the container tag, the 7 W reboot
behaviour, and `detectnet`'s 5×-inflated timer. In every case the value came from checking the
claim against the actual hardware and correcting it.

The study is more correct *because* the assumptions were tested, not accepted. That is the
workflow I'd bring to a team: use AI to move quickly, but treat every claim it makes about the
real system — and every number a tool prints — as a hypothesis to verify.
