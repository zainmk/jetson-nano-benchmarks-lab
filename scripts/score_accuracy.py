#!/usr/bin/env python3
"""
Score the Experiment B class-presence accuracy proxy (Table 5).

Compares detector output against the verified ground truth in
data/clip_ground_truth.csv for the 10 sampled frames.

This is CLASS-PRESENCE scoring, not detection scoring: the ground truth lists
which COCO classes appear in each frame, with no bounding boxes, so localization
is not assessed. Per frame, comparing the SET of ground-truth classes against the
SET of detected classes:
    correct        = class present and reported
    missed         = class present, not reported
    false positive = class reported, not present

Confidence threshold is applied uniformly to every model. jetson-inference's
detectNet defaults to 0.5 while Ultralytics defaults to 0.25, so YOLO output is
re-filtered here to match SSD; otherwise YOLO is credited for detections SSD
would never have emitted.

Run from the repo root:
    python scripts/score_accuracy.py [threshold]
"""
import csv, sys, re
from pathlib import Path
from collections import defaultdict

THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
ROOT = Path(__file__).resolve().parent.parent

# COCO 80-class ids -> names (the subset that appears in this clip's output)
COCO80 = {41: "cup", 42: "fork", 56: "chair", 60: "dining table", 61: "toilet",
          62: "tv", 64: "mouse", 65: "remote", 67: "cell phone", 73: "book",
          79: "toothbrush"}

# ---- ground truth -----------------------------------------------------------
truth = {}
with open(ROOT / "data/clip_ground_truth.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(l for l in f if not l.startswith("#")):
        truth[row["frame"]] = set(o.strip() for o in row["objects"].split(";"))
frames = sorted(truth)

# ---- YOLO: one .txt per frame, "cls xc yc w h conf" -------------------------
def load_yolo(variant):
    det = defaultdict(set)
    conf = defaultdict(list)
    for fr in frames:
        p = ROOT / f"data/yolo_out/{variant}/labels/{fr}.txt"
        if not p.exists():
            continue                     # no detections at all for that frame
        for line in p.read_text().split("\n"):
            parts = line.split()
            if len(parts) < 6:
                continue
            cid, c = int(parts[0]), float(parts[5])
            if c >= THRESH:
                det[fr].add(COCO80.get(cid, f"class{cid}"))
                conf[fr].append(c)
    return det, conf

# ---- SSD: parsed from the detectnet console log ----------------------------
def load_ssd():
    det = defaultdict(set)
    conf = defaultdict(list)
    cur = None
    for line in (ROOT / "data/ssd_detections.log").read_text().split("\n"):
        m = re.match(r"=== (frame_\d+) ===", line)
        if m:
            cur = m.group(1); continue
        m = re.search(r"class #\d+ \((.+?)\)\s+confidence=([0-9.]+)", line)
        if m and cur:
            name, c = m.group(1), float(m.group(2))
            if c >= THRESH:
                det[cur].add(name)
                conf[cur].append(c)
    return det, conf

models = {
    "SSD-Mobilenet-v2": load_ssd(),
    "YOLOv8n":          load_yolo("n"),
    "YOLOv8s":          load_yolo("s"),
}

# ---- score ------------------------------------------------------------------
print(f"Class-presence accuracy @ confidence >= {THRESH}")
print(f"{len(frames)} frames, {sum(len(truth[f]) for f in frames)} ground-truth instances\n")
print(f"{'Model':<20} {'Correct':>8} {'Missed':>8} {'False+':>8} {'Recall':>8} {'Precision':>10} {'MeanConf':>9}")
print("-" * 76)

for name, (det, conf) in models.items():
    correct = missed = fp = 0
    for fr in frames:
        gt, d = truth[fr], det.get(fr, set())
        correct += len(gt & d)
        missed  += len(gt - d)
        fp      += len(d - gt)
    allc = [c for v in conf.values() for c in v]
    recall = correct / (correct + missed) if correct + missed else 0
    prec   = correct / (correct + fp) if correct + fp else 0
    mean_c = sum(allc) / len(allc) if allc else 0
    print(f"{name:<20} {correct:>8} {missed:>8} {fp:>8} {recall:>7.0%} {prec:>10.0%} {mean_c:>9.3f}")

# ---- per-frame detail -------------------------------------------------------
print("\nPer-frame detail:")
for fr in frames:
    print(f"\n  {fr}  truth: {', '.join(sorted(truth[fr]))}")
    for name, (det, _) in models.items():
        d = det.get(fr, set())
        hit  = ", ".join(sorted(truth[fr] & d)) or "-"
        miss = ", ".join(sorted(truth[fr] - d)) or "-"
        bad  = ", ".join(sorted(d - truth[fr])) or "-"
        print(f"    {name:<18} hit: {hit:<28} miss: {miss:<28} FP: {bad}")
