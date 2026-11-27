"""Step 4: build the "is this a scene photo at all?" check used by the app.

The classifier always picks one of the six scenes, even for a photo of a trash can or an emoji.
This script stores 8 "typical photo" reference points per scene (48 in total): k-means centres of
the ResNet50 features of the training photos. In the app, a photo whose features are far from all
48 points does not look like any scene the model was trained on.

Two cut-offs are set from the 10% of training photos held back for validation:
  - unusual:   5% of real scene photos score below this  -> answer shown with a warning
  - not_scene: 1% of real scene photos score below this  -> "Not a scene photo"

Usage: python training/build_scene_check.py data/hf/features_V2.pt scene_check.npz
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

feat_path, out_path = sys.argv[1], sys.argv[2]
K, SEED = 8, 42
torch.manual_seed(SEED)
d = torch.load(feat_path, weights_only=True)
x, xf, y = d["train"]["x"], d["train"]["x_flip"], d["train"]["y"]

# Same stratified 90/10 split as train_head.py, so validation photos were never used here.
g = torch.Generator().manual_seed(SEED)
tr, va = [], []
for c in range(6):
    idx = (y == c).nonzero().squeeze(1)
    idx = idx[torch.randperm(len(idx), generator=g)]
    n = round(len(idx) * 0.1)
    va.append(idx[:n]); tr.append(idx[n:])
tr, va = torch.cat(tr), torch.cat(va)
Xtr, Ytr = torch.cat([x[tr], xf[tr]]), torch.cat([y[tr], y[tr]])

centres = []
for c in range(6):
    f = F.normalize(Xtr[Ytr == c], dim=1)
    C = f[torch.randperm(len(f))[:K]].clone()
    for _ in range(30):  # spherical k-means
        a = (f @ C.T).argmax(1)
        C = F.normalize(torch.stack([f[a == j].mean(0) if (a == j).any() else C[j] for j in range(K)]), dim=1)
    centres.append(C)
centres = torch.cat(centres)

sim_val = (F.normalize(x[va], dim=1) @ centres.T).max(1).values
unusual = torch.quantile(sim_val, 0.05).item()
not_scene = torch.quantile(sim_val, 0.01).item()
sim_test = (F.normalize(d["test"]["x"], dim=1) @ centres.T).max(1).values
print(f"cut-offs: unusual < {unusual:.4f}, not a scene < {not_scene:.4f}")
print(f"test photos flagged: unusual {(sim_test < unusual).float().mean() * 100:.1f}%, "
      f"not a scene {(sim_test < not_scene).float().mean() * 100:.1f}%")

np.savez_compressed(out_path, centres=centres.numpy().astype(np.float16),
                    unusual=np.float32(unusual), not_scene=np.float32(not_scene))
print("saved", out_path)
print(json.dumps({"method": f"cosine similarity to {K} k-means centres per scene (ResNet50 features)",
                  "unusual_below": round(unusual, 4), "not_scene_below": round(not_scene, 4)}))
