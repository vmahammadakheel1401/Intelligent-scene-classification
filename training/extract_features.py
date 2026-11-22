"""Step 1 of CPU training: turn every photo into 2,048 ResNet50 features (done once, the slow part).

Needs data/hf/train.parquet and data/hf/test.parquet (see README, "Retrain on your own computer")
and `pip install pyarrow`.

Usage:  python training/extract_features.py data/hf V2
Writes data/hf/features_V2.pt (train features, mirrored-train features, test features, labels).
Uses exactly the app's preprocessing: Resize((224, 224)) -> ToTensor -> ImageNet Normalize.
"""
import io
import sys
import time

import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import models, transforms

data_dir, wname = sys.argv[1], sys.argv[2]
torch.set_num_threads(max(1, torch.get_num_threads()))
weights = getattr(models.ResNet50_Weights, f"IMAGENET1K_{wname}")
net = models.resnet50(weights=weights)
net.fc = torch.nn.Identity()
net.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run(split, flip_too):
    table = pq.read_table(f"{data_dir}/{split}.parquet")
    imgs = table.column("image").to_pylist()
    labels = torch.tensor(table.column("label").to_pylist())
    feats, feats_flip = [], []
    bs, t0 = 64, time.time()
    for i in range(0, len(imgs), bs):
        batch = torch.stack([tf(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in imgs[i:i + bs]])
        with torch.inference_mode():
            feats.append(net(batch))
            if flip_too:
                feats_flip.append(net(torch.flip(batch, dims=[3])))
        done = min(i + bs, len(imgs))
        if (i // bs) % 10 == 0 or done == len(imgs):
            el = time.time() - t0
            print(f"{split}: {done}/{len(imgs)}  {el:.0f}s  eta {el / done * (len(imgs) - done):.0f}s", flush=True)
    out = {"x": torch.cat(feats), "y": labels}
    if flip_too:
        out["x_flip"] = torch.cat(feats_flip)
    return out


result = {"weights": f"IMAGENET1K_{wname}", "test": run("test", False), "train": run("train", True)}
torch.save(result, f"{data_dir}/features_{wname}.pt")
print("saved", f"{data_dir}/features_{wname}.pt", flush=True)
