"""Flask web app for the Intelligent Scene Image Classification project.

Routes
  GET  /             upload page
  POST /predict      classic form upload (works without JavaScript)
  POST /api/predict  JSON API used by the page's JavaScript
  GET  /about        model details and results
"""
import json
import os
import warnings

import torch
from flask import Flask, jsonify, redirect, render_template, request, url_for
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "natural_scenes_image_resnet_classifier.pt"))
META_PATH = os.environ.get("MODEL_META_PATH", os.path.join(BASE_DIR, "model_meta.json"))

MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
LOW_CONFIDENCE = 0.50  # below this, the page says "not sure"

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_INFO = {
    "buildings": "Built structures fill the frame: skylines, towers, facades.",
    "forest": "Dense trees and greenery, woodland and jungle.",
    "glacier": "Large bodies of ice, ice fields and icebergs.",
    "mountain": "Peaks, ridges and rocky slopes, often with snow.",
    "sea": "Open water, beaches, coastlines and waves.",
    "street": "Roads and lanes seen from street level, with buildings alongside.",
}
CLASS_THUMBS = {  # one bundled sample photo that represents each class
    "buildings": "test_image_4.jpg", "forest": "test_image_12.jpg", "glacier": "test_image_10.jpg",
    "mountain": "test_image_3.jpg", "sea": "test_image_7.jpg", "street": "test_image_5.jpg",
}
SAMPLE_LABELS = {  # what each bundled sample image shows (checked by eye)
    "test_image_1.jpg": "glacier", "test_image_2.jpg": "forest", "test_image_3.jpg": "mountain",
    "test_image_4.jpg": "buildings", "test_image_5.jpg": "street", "test_image_6.jpg": "buildings",
    "test_image_7.jpeg": "sea", "test_image_8.jpg": "sea", "test_image_9.jpg": "glacier",
    "test_image_10.jpg": "glacier", "test_image_11.jpg": "mountain", "test_image_12.jpeg": "forest",
}


class UserError(Exception):
    """A problem with the upload that the user can fix (shown as a friendly message)."""


def load_meta():
    """model_meta.json is written by the training notebook. Missing file = old model."""
    meta = {"img_size": 128, "classes": CLASSES, "version": "v1 (original)"}
    try:
        with open(META_PATH, encoding="utf-8") as f:
            meta.update(json.load(f))
    except (OSError, ValueError):
        pass
    return meta


def build_resnet(num_classes):
    """Same architecture as the training notebook (ResNet50 + small head)."""
    net = models.resnet50(weights=None)
    net.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.3),
        nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
        nn.Linear(128, num_classes), nn.LogSoftmax(dim=1),
    )
    return net


def load_model(path, num_classes):
    """Loads either a weights-only file (new notebook) or a whole pickled model (original file)."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)  # safe loader
    except FileNotFoundError:
        raise
    except Exception:
        # The original file is a whole pickled model, which the safe loader refuses.
        # Only load it this way because it is our own file.
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        net = build_resnet(num_classes)
        net.load_state_dict(obj.get("state_dict", obj))  # fp16 weights are cast to fp32 here
    else:
        net = obj
    return net.float().eval()


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

meta = load_meta()
CLASSES = meta.get("classes", CLASSES)
transform = transforms.Compose([
    transforms.Resize(size=(meta["img_size"], meta["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    model = load_model(MODEL_PATH, len(CLASSES)).to(device)
    model_load_error = None
except Exception as exc:  # the page still loads and explains the problem
    model = None
    model_load_error = f"{type(exc).__name__}: {exc}"
    print(f"[app] Model could not be loaded from {MODEL_PATH}: {model_load_error}")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def classify(file_storage):
    """Validates an uploaded file and returns a result dict. Raises UserError for bad input."""
    if model is None:
        raise RuntimeError("The model is not available on the server right now.")
    if file_storage is None or not file_storage.filename:
        raise UserError("Please choose an image first.")
    ext = file_storage.filename.rsplit(".", 1)[-1].lower() if "." in file_storage.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UserError("That file type isn't supported. Please upload a JPG, PNG, WEBP, BMP or GIF image.")
    try:
        image = Image.open(file_storage.stream)
        image = ImageOps.exif_transpose(image).convert("RGB")  # respect phone photo rotation
    except Exception:
        raise UserError("We couldn't read that file as an image. It may be damaged or not really an image.")

    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).exp()[0].cpu()  # the model ends in LogSoftmax, so exp() gives probabilities
    top = torch.topk(probs, k=3)
    top3 = [{"label": CLASSES[i], "probability": round(float(p), 4)} for p, i in zip(top.values, top.indices)]
    best = top3[0]
    return {
        "ok": True,
        "label": best["label"],
        "confidence": best["probability"],
        "low_confidence": best["probability"] < LOW_CONFIDENCE,
        "description": CLASS_INFO.get(best["label"], ""),
        "top3": top3,
        "probabilities": [{"label": c, "probability": round(float(p), 4)} for c, p in zip(CLASSES, probs)],
        "filename": file_storage.filename,
    }


def page_context(**extra):
    samples = [{"file": f, "thumb": os.path.splitext(f)[0] + ".jpg", "label": lbl} for f, lbl in SAMPLE_LABELS.items()]
    classes = [{"name": c, "info": CLASS_INFO.get(c, ""), "thumb": CLASS_THUMBS.get(c)} for c in CLASSES]
    ctx = dict(classes=classes, samples=samples,
               max_upload_mb=MAX_UPLOAD_MB, low_confidence=LOW_CONFIDENCE, model_ready=model is not None,
               model_load_error=model_load_error, meta=meta)
    ctx.update(extra)
    return ctx


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", **page_context())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":  # someone opened /predict directly or refreshed
        return redirect(url_for("home"))
    try:
        result = classify(request.files.get("image"))
        return render_template("index.html", **page_context(result=result))
    except UserError as err:
        return render_template("index.html", **page_context(error=str(err))), 400
    except RuntimeError as err:
        return render_template("index.html", **page_context(error=str(err))), 503


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        return jsonify(classify(request.files.get("image")))
    except UserError as err:
        return jsonify(ok=False, error=str(err)), 400
    except RuntimeError as err:
        return jsonify(ok=False, error=str(err)), 503


@app.route("/about")
def about():
    return render_template("about.html", **page_context())


@app.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.svg"))


def wants_json():
    return request.path.startswith("/api/")


@app.errorhandler(413)
def too_large(_err):
    msg = f"That file is too big. Please upload an image smaller than {MAX_UPLOAD_MB} MB."
    if wants_json():
        return jsonify(ok=False, error=msg), 413
    return render_template("index.html", **page_context(error=msg)), 413


@app.errorhandler(404)
def not_found(_err):
    if wants_json():
        return jsonify(ok=False, error="Not found."), 404
    return render_template("error.html", code=404, message="We couldn't find that page."), 404


@app.errorhandler(500)
def server_error(_err):
    msg = "Something went wrong on our side. Please try again."
    if wants_json():
        return jsonify(ok=False, error=msg), 500
    return render_template("error.html", code=500, message=msg), 500


if __name__ == "__main__":
    # Debug mode shows code and file paths on errors, so it is OFF unless you ask for it:
    #   Windows PowerShell:  $env:FLASK_DEBUG="1"; python app.py
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=os.environ.get("FLASK_DEBUG") == "1")
