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

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, redirect, render_template, request, url_for
from PIL import Image, ImageOps

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ONNX version of the trained model (made by training/export_onnx.py). ONNX Runtime is ~80 MB installed,
# PyTorch is over 1 GB, which is why the app can run on Vercel.
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "model.onnx"))
META_PATH = os.environ.get("MODEL_META_PATH", os.path.join(BASE_DIR, "model_meta.json"))

MAX_UPLOAD_MB = 10  # largest file the server accepts
MAX_PICK_MB = 25  # largest photo the page accepts; it shrinks big photos before uploading
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
COMPARISON_GAP = 0.15
SIMILAR_CLASS_GUIDANCE = {
    frozenset(("glacier", "mountain")):
        "Look for broad ice fields or icebergs (glacier) versus rocky peaks or ridges (mountain).",
    frozenset(("buildings", "street")):
        "Look for structures dominating the frame (buildings) versus a road or street scene (street).",
}
CLASS_THUMBS = {  # one bundled sample photo that represents each class
    "buildings": "test_image_4.jpg", "forest": "test_image_12.jpg", "glacier": "test_image_10.jpg",
    "mountain": "test_image_3.jpg", "sea": "test_image_8.jpg", "street": "test_image_5.jpg",
}
SAMPLE_LABELS = {  # what each bundled sample image shows (checked by eye)
    "test_image_1.jpg": "glacier", "test_image_2.jpg": "forest", "test_image_3.jpg": "mountain",
    "test_image_4.jpg": "buildings", "test_image_5.jpg": "street", "test_image_6.jpg": "buildings",
    # test_image_7.jpeg is left out: it carries an "Unsplash+" (paid licence) watermark.
    "test_image_8.jpg": "sea", "test_image_9.jpg": "glacier",
    "test_image_10.jpg": "glacier", "test_image_11.jpg": "mountain", "test_image_12.jpeg": "forest",
}


class UserError(Exception):
    """A problem with the upload that the user can fix (shown as a friendly message)."""


def compare_top_classes(top3):
    if len(top3) < 2:
        return None
    first, second = top3[:2]
    guidance = SIMILAR_CLASS_GUIDANCE.get(frozenset((first["label"], second["label"])))
    if guidance is None or first["probability"] - second["probability"] > COMPARISON_GAP:
        return None
    return {
        "first_label": first["label"],
        "first_probability": first["probability"],
        "second_label": second["label"],
        "second_probability": second["probability"],
        "guidance": guidance,
    }


def load_meta():
    """model_meta.json is written by the training notebook. Missing file = old model."""
    meta = {"img_size": 128, "classes": CLASSES, "version": "v1 (original)"}
    try:
        with open(META_PATH, encoding="utf-8") as f:
            meta.update(json.load(f))
    except (OSError, ValueError):
        pass
    return meta


def load_model(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))  # 0 = use all cores
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


meta = load_meta()
CLASSES = meta.get("classes", CLASSES)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image):
    """Same steps as training: resize to a square, scale to 0-1, ImageNet normalisation, CHW."""
    size = meta["img_size"]
    arr = np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None]  # shape (1, 3, H, W)


try:
    model = load_model(MODEL_PATH)
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

    log_probs = model.run(None, {model.get_inputs()[0].name: preprocess(image)})[0][0]
    probs = np.exp(log_probs)  # the model ends in LogSoftmax, so exp() gives probabilities
    top3 = [{"label": CLASSES[i], "probability": round(float(probs[i]), 4)} for i in np.argsort(-probs)[:3]]
    best = top3[0]
    return {
        "ok": True,
        "label": best["label"],
        "confidence": best["probability"],
        "low_confidence": best["probability"] < LOW_CONFIDENCE,
        "description": CLASS_INFO.get(best["label"], ""),
        "top3": top3,
        "comparison": compare_top_classes(top3),
        "probabilities": [{"label": c, "probability": round(float(p), 4)} for c, p in zip(CLASSES, probs)],
        "filename": file_storage.filename,
    }


def page_context(**extra):
    samples = [{"file": f, "thumb": os.path.splitext(f)[0] + ".jpg", "label": lbl} for f, lbl in SAMPLE_LABELS.items()]
    classes = [{"name": c, "info": CLASS_INFO.get(c, ""), "thumb": CLASS_THUMBS.get(c)} for c in CLASSES]
    ctx = dict(classes=classes, samples=samples,
               max_upload_mb=MAX_UPLOAD_MB, max_pick_mb=MAX_PICK_MB, low_confidence=LOW_CONFIDENCE, model_ready=model is not None,
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
