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

## 11. A prediction that failed — the bottleneck never flipped

**Zain:** If YOLOv8s is heavy enough, does the GPU become the bottleneck instead of the camera?

**Claude:** That's the interesting contrast to look for — SSD is camera-bound, but a 640² model
might drop below 30 fps and flip it.

**Zain:** *(runs the sweep)* `throughput=159.327fps`

**Claude:** That refutes the prediction. YOLOv8s runs **5× faster** than the camera can deliver
frames — still camera-bound. And at the *lowest* power mode it holds 45.7 fps, still 1.5× the
requirement. **No configuration tested on this platform is compute-bound under a 30 fps webcam.**

**Design decision:** Rather than bury the failed prediction, I wrote it into the discussion as a
stronger result than the one expected: the bottleneck is not merely present for lightweight models,
it is *universal* across everything the board can reasonably run. Locating the actual crossover —
where a model finally cannot keep up — became the lead item in future work.

---

## 12. Measuring the inversion, and catching three errors doing it

**Zain:** *(after the power sweep)* mean_power_W=3.97 at 7 W.

**Claude:** Energy per frame falls monotonically as the power mode drops — 212 → 184 → **137 mJ**
for identical 29 fps output. MAXN burns 1.55× the energy to produce exactly the same result. This
inverts the Experiment A conclusion.

Three corrections came out of scrutinising that result rather than accepting it:

1. **My own pre-measurement estimate was wrong.** I had predicted the modes would land within a few
   percent, assuming a mode-independent idle floor. Idle power itself scales down with the mode, so
   the effect was far larger than predicted. The wrong prediction is recorded in the report, because
   the same reasoning would have understated low-power operation in a design review.
2. **The GPU-utilisation column was mislabelled.** I had written 77% / 49% / 24% as evidence of duty
   cycle; those are *peak* values from `max(GR3D_FREQ)`. Mean utilisation is ~3%. The mechanism I had
   described was not supported by that column and had to be replaced.
3. **The measurement windows were misaligned.** Cross-referencing host log timestamps against the
   harness completion time showed the 7 W log ended 19 s *before* its run finished while the 15 W log
   ran 113 s *past* it — loose in opposite directions.

**Zain:** Is it possible the delay between starting/stopping tegrastats and the container is
offsetting the data?

**Design decision:** That question was the right one and it drove the audit. The corrected mechanism
turned out to be *cleaner* than the original: segmenting by GPU activity showed active-sample power
is indistinguishable from idle-sample power, meaning board draw is a **static function of the power
mode** rather than of the inference. The direction of the result survives the window problem (the
bias would only widen the gap), but the absolute values are reported as approximate, with the
methodology flaw documented in §8 rather than quietly smoothed over.

---

## 13. "Is this actually novel, or am I stating the obvious?"

**Zain:** It seems like the experiment is just identifying that the USB camera is the bottleneck —
is this a strong enough lab report? Be straightforward.

**Claude:** The conclusions are not novel. That reduced clocks lower power, and that a slow sensor
bottlenecks a fast accelerator, are established results; scaling frequency to just meet a deadline is
textbook real-time systems practice. If the report is framed as *discovering* them, an experienced
reviewer will read it as naive.

What is defensible is the characterization and the methodology — measuring the magnitudes precisely
on this platform, and documenting what it takes to obtain numbers worth trusting.

**Design decision:** Reframed the entire report around that honest positioning. The title, abstract,
and contributions were rewritten from an effects study to a measurement study, opening with an
explicit statement that several conclusions are confirmatory rather than novel. The documented
measurement pitfalls — a tool over-reporting latency 5×, a 3× capture collapse from a pixel format,
power-mode IDs contradicting vendor documentation — were promoted from footnotes to a headline
contribution, because they are the part a reader cannot get anywhere else.

---

## 14. Catching an unfair comparison before it reached a table

**Claude:** *(scoring the detection accuracy)* Before these numbers go in — the thresholds don't
match. `detectNet` defaults to 0.5, Ultralytics to 0.25. YOLO was credited for detections SSD would
never have emitted; several of its "false positives" sit below SSD's cutoff.

Re-scoring YOLO at 0.5 moved YOLOv8n from 41% to 31% recall and 5 false positives to 2. **The model
ordering held at both thresholds**, so the ranking is robust — but the magnitudes were not
comparable until the thresholds were matched.

**Design decision:** Wrote a scoring script (`score_accuracy.py`) that applies a single threshold
uniformly and takes it as an argument, so threshold sensitivity is checkable rather than assumed.
Also flagged and corrected a ground-truth error of my own: three independent detectors reported a
`dining table` I had omitted from the labels, which on review was simply the desk — a reminder that
in AI-assisted annotation, label errors propagate identically into every model's score.

---

## 15. Free data — the storage dimension

**Zain:** Do we have any assessment of the storage difference between FP32, FP16, INT8?

**Claude:** We do, in the archived logs — `trtexec` records engine size and we never extracted it.

Serialized engine size halves at each precision step, giving a consistent **3.9–4.0× reduction from
FP32 to INT8** across all three architectures. VGG-16 goes from 528 MiB to 133 MiB. And among the
detectors: SSD-Mobilenet-v2 has the *largest* engine (36 MiB) and the *worst* accuracy, while
YOLOv8s is 33% smaller with 2.3× the recall.

**Design decision:** Added the storage axis to the report without any new measurement — the data was
already archived. It also sharpened an omission: every measured cost of INT8 is now lower (time,
energy, storage), leaving accuracy as the single unquantified one, which frames Table 3's absence as
a specific known gap rather than a vague hole.

---

## Reflection

Two things stand out across this project.

**The AI's confident claims about the hardware were frequently wrong.** The power-mode IDs, the
ambient-temperature source, the container tag, the 7 W reboot behaviour, the predicted energy
parity, and my own mislabelled utilisation column — each was corrected only by checking against the
device. So were the *tools'* own numbers: `detectnet` over-reported inference latency by 5×, and two
detectors were nearly compared at mismatched confidence thresholds. The measurements that survived
are the ones that were doubted first.

**The most useful question I asked was whether the findings were actually interesting.** They largely
are not, in the sense of novelty — reduced clocks save power, slow sensors bottleneck fast chips.
Recognising that changed the report from a discovery claim into a characterization and methodology
study, which is both more honest and more defensible. The final result does carry a real design
insight — that the optimal power mode *and* the optimal model size both invert once the sensor rather
than the accelerator sets the pace — but it is presented as a consequence of careful measurement
rather than as a surprise.

That is the workflow I would bring to a team: use AI to move quickly, verify every claim it makes
about the real system, and be willing to ask whether the result is worth the write-up before writing
it up.
