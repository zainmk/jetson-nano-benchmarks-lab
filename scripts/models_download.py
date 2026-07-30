# PYTHON ASSIST FILE TO EXPORT ONNX MODELS FROM TORCHVISION
#
# Purpose: produce the Track A classification models as ONNX for the Experiment A
#          trtexec timing runs. Run on the HOST, then scp the .onnx files to the Orin.
#
# Setup (Windows PowerShell, from the repo root):
#   .\.venv\Scripts\Activate.ps1
#   python -m pip install --upgrade pip
#   pip install torch torchvision      # CPU build is sufficient for export
#   pip install onnx onnxscript        # Dependency for onnx export

#
# Run:
#   python models_download.py
#
# Then copy to the Jetson via scp:
#   scp resnet18.onnx resnet50.onnx vgg16.onnx <user>@<jetson-ip>:~/models/

import torch, torchvision

# Track A models (A2, A3, A5). All use 224x224 input; batch size fixed at 1.
#
# The trailing number = network depth (count of learnable layers). More depth/params
# generally means more compute + memory per inference, which is the scaling axis RQ1 tests.
#
#   resnet18 / resnet50 : ResNet uses residual "skip connections" so very deep nets stay
#                         trainable. Very parameter-EFFICIENT (ResNet-50 ~26M params beats
#                         VGG-16 despite ~5x fewer weights). Two depths give a clean
#                         scaling ladder: 18 (light) -> 50 (mid).
#   vgg16               : Older "just stack 3x3 convs + huge fully-connected layers" design.
#                         Only 16 layers but ~138M params -> parameter/memory-BANDWIDTH heavy,
#                         not depth-heavy. The deliberate outlier "heavy baseline": tests
#                         whether latency/energy track FLOPs or memory bandwidth on the Orin.
#
# Weights: use IMAGENET1K_V1 for ALL three (ResNet-50 also offers a better-trained V2, but
# 18 and VGG-16 do not). Keeping the same weight version across models is a CONTROLLED
# variable so the Table 3 accuracy comparison stays fair.

# This model 
# set allows us to assess the hardware performance scaling with model size, while keeping the task (ImageNet classification) and input size (224x224) constant.

models = {
    "resnet18": torchvision.models.resnet18,   # A2 — small residual (~12M params)
    "resnet50": torchvision.models.resnet50,   # A3 — mid residual   (~26M params)
    "vgg16":    torchvision.models.vgg16,       # A5 — heavy baseline (~138M params)
}

for name, ctor in models.items():
    m = ctor(weights="IMAGENET1K_V1").eval()
    torch.onnx.export(
        m,
        torch.randn(1, 3, 224, 224),           # dummy input; fixes the input shape
        f"{name}.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    print(f"exported {name}.onnx")
