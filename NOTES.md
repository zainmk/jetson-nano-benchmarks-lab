# Working notes — candidate future work

*Not part of the lab report. Ideas and observations parked here for later consideration;
nothing below has been measured or written into `lab.md`.*

---

## Clock policy: the benchmark config is not the deployment config

**Observation.** The Jetson already implements the between-frame power saving we would
otherwise try to hand-code. `/sys/devices/platform/gpu.0/power/control` set to `auto`
lets the driver runtime-suspend the GPU when idle, and DVFS scales clocks down under
light load. Both are visible in the `nvpmodel -p --verbose` output
(`GPU_POWER_CONTROL_ENABLE / _DISABLE`).

**The catch.** `sudo jetson_clocks` — run before every benchmark in this study for
timing reproducibility (§5.4) — deliberately **pins clocks to maximum and disables that
runtime power management**. So under a sensor-capped pipeline the GPU is being forced to
hold max clocks through ~27 ms of idle per frame, every frame. The board's own power
saving is present but overridden *by the measurement methodology*.

**Implication.** The optimal *deployment* configuration may differ from the optimal
*benchmark* configuration on a third axis beyond model and power mode:

| Knob | Benchmark setting | Likely deployment setting |
|---|---|---|
| Power mode | fixed per run | lowest that meets the deadline (7 W, measured) |
| Model | varied | lightest meeting accuracy (pending) |
| **Clock policy** | **pinned (`jetson_clocks`)** | **unpinned — let DVFS / runtime PM scale down** |

**Candidate experiment (unmeasured).** Run the camera pipeline at a fixed 30 fps in one
power mode, twice: (a) with `jetson_clocks` pinned, (b) without it, so DVFS and
`power/control=auto` are active. Compare mean board power from `tegrastats`. A meaningful
drop in (b) at identical 30 fps output would show that pinning clocks — correct for
measurement — is wasteful for deployment.

**Caveats to state if pursued.**
- Unpinned clocks widen latency variance (p95/p99), trading deadline margin for power.
  With 1.5× headroom at 7 W this is probably acceptable, but it is a trade-off, not free.
- Deep sleep (SC7) is **not** viable at frame granularity — wake latency is orders of
  magnitude beyond the 33 ms budget. Only shallow GPU suspend / DVFS fit in the gap.
- Gains are bounded by the whole-board idle floor (~3.3 W measured): LPDDR5 refresh, PMIC
  losses, USB + camera, PHY, fan. The GPU is only one slice of that, so perfect gating
  would not approach zero.

**Status.** Not measured. Would require methodology distinct from the existing runs (all
current data was collected with clocks pinned), so it should be reported as a separate
comparison rather than mixed into Tables 1–6.
