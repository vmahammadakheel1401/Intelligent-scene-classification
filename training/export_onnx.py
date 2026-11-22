"""Step 3: convert the trained PyTorch model (.pt) to ONNX so the web app runs without PyTorch.

The app uses ONNX Runtime (about 80 MB installed) instead of PyTorch (over 1 GB), which is what lets it
fit on Vercel. Weights are stored as float16 to halve the file (about 49 MB) and are turned back
into float32 when the app loads the model, so predictions match the .pt file.

Needs: pip install -r requirements-train.txt
Usage: python training/export_onnx.py natural_scenes_image_resnet_classifier.pt model.onnx
"""
import sys

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from torch import nn
from torchvision import models

src, dst = sys.argv[1], sys.argv[2]

net = models.resnet50(weights=None)
net.fc = nn.Sequential(
    nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.3),
    nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
    nn.Linear(128, 6), nn.LogSoftmax(dim=1),
)
state = torch.load(src, map_location="cpu", weights_only=True)
net.load_state_dict(state.get("state_dict", state))
net = net.float().eval()

tmp = dst + ".fp32.onnx"
torch.onnx.export(net, torch.zeros(1, 3, 224, 224), tmp, input_names=["image"], output_names=["log_probs"],
                  dynamic_axes={"image": {0: "batch"}, "log_probs": {0: "batch"}}, opset_version=17, dynamo=False)

# Store every large float32 weight as float16 + a Cast node back to float32.
model = onnx.load(tmp)
g = model.graph
new_inits, casts = [], []
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    if arr.dtype == np.float32 and arr.size > 16:
        half = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
        new_inits.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT))
    else:
        new_inits.append(init)
del g.initializer[:]
g.initializer.extend(new_inits)
nodes = casts + list(g.node)
del g.node[:]
g.node.extend(nodes)
onnx.checker.check_model(model)
onnx.save(model, dst)

import os  # noqa: E402
os.remove(tmp)
print("saved", dst, f"{os.path.getsize(dst) / 1e6:.1f} MB")
