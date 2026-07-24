# PYTHON ASSIST FILE TO EXPORT ONNX MODELS FROM TORCHVISION
#
# Purpose: produce the Track A classification models as ONNX for the Experiment A
#          trtexec timing runs. Run on the HOST, then scp the .onnx files to the Orin.
#
# Setup (Windows PowerShell, from the repo root):
#   .\.venv\Scripts\Activate.ps1
#   python -m pip install --upgrade pip
#   pip install torch torchvision      # CPU build is sufficient for export
#
# Run:
#   python models_download.py
#
# Then copy to the Jetson via scp:
#   scp resnet18.onnx resnet50.onnx vgg16.onnx <user>@<jetson-ip>:~/models/

import torch, torchvision

# Track A models (A2, A3, A5). All use 224x224 input; batch size fixed at 1.
models = {
    "resnet18": torchvision.models.resnet18,   # A2 — small residual
    "resnet50": torchvision.models.resnet50,   # A3 — mid residual
    "vgg16":    torchvision.models.vgg16,       # A5 — heavy baseline
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
