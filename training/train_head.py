"""Step 2 of CPU training: train the classifier head on the saved features (takes seconds),
test it on the 3,000 test photos, and with --save write the app's model file and model_meta.json.

Usage:  python training/train_head.py data/hf/features_V2.pt . --save
"""
import datetime
import json
import sys

import torch
from torch import nn
from torchvision import models

feat_path, out_dir = sys.argv[1], sys.argv[2]
SAVE = "--save" in sys.argv
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
SEED, EPOCHS, BATCH, LR, WD, SMOOTH = 42, 40, 256, 1e-3, 1e-4, 0.1

torch.manual_seed(SEED)
d = torch.load(feat_path, weights_only=True)
x, xf, y = d["train"]["x"], d["train"]["x_flip"], d["train"]["y"]
xt, yt = d["test"]["x"], d["test"]["y"]

# Stratified 90/10 train/validation split of the training folder (test set stays untouched).
g = torch.Generator().manual_seed(SEED)
tr_idx, va_idx = [], []
for c in range(len(CLASSES)):
    idx = (y == c).nonzero().squeeze(1)
    idx = idx[torch.randperm(len(idx), generator=g)]
    n_va = round(len(idx) * 0.1)
    va_idx.append(idx[:n_va]); tr_idx.append(idx[n_va:])
tr_idx, va_idx = torch.cat(tr_idx), torch.cat(va_idx)


def make_head():
    return nn.Sequential(
        nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.3),
        nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
        nn.Linear(128, len(CLASSES)), nn.LogSoftmax(dim=1),
    )


head = make_head()
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.NLLLoss()


def smoothed_nll(logp, target):
    # label smoothing on log-probabilities
    return (1 - SMOOTH) * loss_fn(logp, target) + SMOOTH * (-logp.mean(dim=1)).mean()


def accuracy(xs, ys):
    head.eval()
    with torch.no_grad():
        return (head(xs).argmax(1) == ys).float().mean().item()


best_val, best_state, best_epoch, history = 0.0, None, 0, []
for epoch in range(1, EPOCHS + 1):
    head.train()
    perm = tr_idx[torch.randperm(len(tr_idx))]
    total = 0.0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]
        flip = torch.rand(len(b)) < 0.5  # augmentation: original or mirrored photo
        xb = torch.where(flip[:, None], xf[b], x[b])
        loss = smoothed_nll(head(xb), y[b])
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(b)
    sched.step()
    va = accuracy(x[va_idx], y[va_idx])
    history.append({"epoch": epoch, "train_loss": round(total / len(perm), 4), "val_accuracy": round(va, 4)})
    if va > best_val:
        best_val, best_epoch = va, epoch
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
    print(f"epoch {epoch:2d}  loss {total / len(perm):.4f}  val {va:.4f}", flush=True)

head.load_state_dict(best_state)
head.eval()
with torch.no_grad():
    pred = head(xt).argmax(1)
test_acc = (pred == yt).float().mean().item()
cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
for t, p in zip(yt, pred):
    cm[t, p] += 1
per_class = {}
for i, c in enumerate(CLASSES):
    tp = cm[i, i].item(); fp = cm[:, i].sum().item() - tp; fn = cm[i].sum().item() - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": cm[i].sum().item()}
print(f"\nbest epoch {best_epoch}  val {best_val:.4f}  TEST {test_acc:.4f}")
print(cm)
for c, r in per_class.items():
    print(f"{c:10s} P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}")

if SAVE:
    weights = getattr(models.ResNet50_Weights, d["weights"])
    net = models.resnet50(weights=weights)
    net.fc = make_head()
    net.fc.load_state_dict(best_state)
    net.eval()
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in net.state_dict().items()}
    torch.save(state, f"{out_dir}/natural_scenes_image_resnet_classifier.pt")
    meta = {
        "version": "v2, head trained on CPU",
        "trained_at": datetime.date.today().isoformat(),
        "img_size": 224,
        "classes": CLASSES,
        "test_accuracy": round(test_acc, 4),
        "test_images": len(yt),
        "best_val_accuracy": round(best_val, 4),
        "epochs_trained": EPOCHS,
        "best_epoch": best_epoch,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "trainable": f"backbone frozen ({d['weights']} weights), only the classifier head trained",
        "augmentation": "Horizontal flips (features of each photo and its mirror image)",
        "optimizer": f"AdamW, learning rate {LR}, weight decay {WD}, cosine schedule, label smoothing {SMOOTH}, {EPOCHS} epochs, best validation epoch kept",
        "train_images": len(tr_idx),
        "val_images": len(va_idx),
        "dataset_source": "Intel Image Classification (Hugging Face mirror sfarrukhm/intel-image-classification)",
        "history": history,
    }
    with open(f"{out_dir}/model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("saved model and meta to", out_dir)
