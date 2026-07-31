# PYTHON ASSIST FILE — export all host-side ONNX models for the lab.
#
# Two model tracks, both exported on the HOST then scp'd to the Orin:
#   Track A (classification, torchvision) — ResNet-18/50, VGG-16
#   Track B (detection, ultralytics)      — YOLOv8n, YOLOv8s
#
# WHERE EACH MODEL IS USED IN THE LAB:
#   resnet18 (A2) — Exp A Tables 1–2 (trtexec timing) + Table 3 (imagenet accuracy)
#   resnet50 (A3) — Exp A Tables 1–2 + Table 3   (also the H5 power-mode sweep, Table 2)
#   vgg16    (A5) — Exp A Tables 1–2 + Table 3
#   yolov8n  (B3) — Exp B Table 4 (detection): Phase 1 infer-only via trtexec vs the 30 fps camera
#   yolov8s  (B4) — Exp B Table 4: the heavier model that may flip the bottleneck GPU→camera (H4)
#
# Setup (Windows PowerShell, from the repo root):
#   .\.venv\Scripts\Activate.ps1
#   python -m pip install --upgrade pip
#   pip install torch torchvision      # CPU build is sufficient for export
#   pip install onnx onnxscript        # dependency for torch.onnx.export
#   pip install ultralytics            # for the YOLOv8 detectors (Track B)
#
# Run:
#   python scripts/models_download.py
#
# Then copy to the Jetson via scp (include any .onnx.data files if produced):
#   scp resnet18.onnx resnet50.onnx vgg16.onnx yolov8n.onnx yolov8s.onnx <user>@<jetson-ip>:~/models/

import torch, torchvision

# ---- Track A — classification (Experiment A) --------------------------------
# The trailing number = network depth (count of learnable layers). More depth/params
# generally means more compute + memory per inference — the scaling axis RQ1 tests.
#   resnet18 / resnet50 : ResNet uses residual "skip connections" so very deep nets stay
#                         trainable. Parameter-EFFICIENT (ResNet-50 ~26M params beats VGG-16
#                         despite ~5x fewer weights). 18 (light) -> 50 (mid) = a scaling ladder.
#   vgg16               : Older "stack 3x3 convs + huge fully-connected layers" design. Only
#                         16 layers but ~138M params -> parameter / memory-BANDWIDTH heavy,
#                         not depth-heavy. The deliberate outlier "heavy baseline".
# Weights: IMAGENET1K_V1 for ALL three — a CONTROLLED variable so the Table 3 accuracy
# comparison stays fair. Task (ImageNet) and input size (224x224) held constant.
classifiers = {
    "resnet18": torchvision.models.resnet18,   # A2 — small residual (~12M params)
    "resnet50": torchvision.models.resnet50,   # A3 — mid residual   (~26M params)
    "vgg16":    torchvision.models.vgg16,       # A5 — heavy baseline (~138M params)
}
for name, ctor in classifiers.items():
    m = ctor(weights="IMAGENET1K_V1").eval()
    torch.onnx.export(
        m,
        torch.randn(1, 3, 224, 224),           # dummy input; fixes the input shape
        f"{name}.onnx",
        input_names=["input"], output_names=["output"], opset_version=13,
    )
    print(f"exported {name}.onnx  (classification, 224x224)")

# ---- Track B — detection (Experiment B) -------------------------------------
# YOLOv8 is NOT a torchvision model, so it exports via Ultralytics (different tool, same
# idea: pretrained -> ONNX). 640x640 input. These feed Exp B Table 4:
#   Phase 1 measures infer-only FPS via trtexec (like Exp A) and compares to the ~30 fps
#   camera ceiling. SSD-Mobilenet was camera-bound (240 fps ≫ 30); YOLOv8s is heavier and
#   may drop below 30 fps -> GPU-bound, the deliberate H4 contrast.
# Wrapped in try/except so the Track A exports above don't require ultralytics.
try:
    from ultralytics import YOLO
    for name in ["yolov8n", "yolov8s"]:          # B3 (tiny), B4 (small)
        YOLO(f"{name}.pt").export(format="onnx", opset=13, imgsz=640)
        print(f"exported {name}.onnx  (detection, 640x640)")
except ImportError:
    print("ultralytics not installed — skipping YOLOv8 export. "
          "Run: pip install ultralytics")

print("done — scp the .onnx files (+ any .onnx.data) to ~/models/ on the Jetson")
