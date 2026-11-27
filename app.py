"""Flask web app for the Intelligent Scene Image Classification project.

Routes
  GET  /             upload page
  POST /predict      classic form upload (works without JavaScript)
  POST /api/predict  JSON API used by the page's JavaScript
  POST /api/predict-batch  up to 8 photos per request (used by the photo organizer)
  GET  /organize     photo organizer: sort a folder of photos into scene folders
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
# "Is this a scene photo at all?" check (made by training/build_scene_check.py).
SCENE_CHECK_PATH = os.environ.get("SCENE_CHECK_PATH", os.path.join(BASE_DIR, "scene_check.npz"))

MAX_UPLOAD_MB = 10  # largest file the server accepts
MAX_PICK_MB = 25  # largest photo the page accepts; it shrinks big photos before uploading
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
LOW_CONFIDENCE = 0.50  # below this, the page says "not sure"
MAX_BATCH = 8  # photos per /api/predict-batch request (the organizer shrinks them first)
MAX_ORGANIZE = 200  # photos per organizer run

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_INFO = {
    "buildings": "Built structures fill the frame: skylines, towers, facades.",
    "forest": "Dense trees and greenery, woodland and jungle.",
    "glacier": "Large bodies of ice, ice fields and icebergs.",
    "mountain": "Peaks, ridges and rocky slopes, often with snow.",
    "sea": "Open water, beaches, coastlines and waves.",
    "street": "Roads and lanes seen from street level, with buildings alongside.",
}
COMPARISON_GAP = 0.15  # top two this close (for a look-alike pair) = "close call"
MIXED_MIN = 0.20  # second scene at least this likely = "mixed scene"
SIMILAR_CLASS_GUIDANCE = {
    frozenset(("glacier", "mountain")):
        "Look for broad ice fields or icebergs (glacier) versus rocky peaks or ridges (mountain).",
    frozenset(("buildings", "street")):
        "Look for structures dominating the frame (buildings) versus a road or street scene (street).",
}
MIXED_GUIDANCE = {  # shown when a photo really contains two scenes (e.g. a building covered in trees)
    frozenset(("buildings", "forest")):
        "The photo has both built structures and a lot of greenery, such as a building covered in plants or houses in a wood.",
    frozenset(("forest", "mountain")): "The photo shows both trees and mountain slopes.",
    frozenset(("glacier", "sea")): "The photo shows both ice and open water, such as icebergs in the sea.",
    frozenset(("mountain", "sea")): "The photo shows both mountains and the sea, such as a rocky coastline.",
    frozenset(("buildings", "sea")): "The photo shows both buildings and water, such as a waterfront.",
    frozenset(("forest", "street")): "The photo shows both a road and a lot of trees, such as a tree-lined street.",
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
    """Close call between look-alike scenes, or a photo that mixes two scenes; None otherwise."""
    if len(top3) < 2:
        return None
    first, second = top3[:2]
    pair = frozenset((first["label"], second["label"]))
    a, b = first["label"].capitalize(), second["label"].capitalize()
    if pair in SIMILAR_CLASS_GUIDANCE and first["probability"] - second["probability"] <= COMPARISON_GAP:
        kind, title, guidance = "close_call", f"Close call: {a} or {b}?", SIMILAR_CLASS_GUIDANCE[pair]
    elif second["probability"] >= MIXED_MIN:
        kind, title = "mixed", f"Mixed scene: {a} and {b}"
        guidance = MIXED_GUIDANCE.get(pair, f"The photo shows signs of both {first['label']} and {second['label']}.")
    else:
        return None
    return {
        "kind": kind,
        "title": title,
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


def load_scene_check(path):
    """Reference points + cut-offs for the scene check. Missing file = check switched off."""
    try:
        data = np.load(path)
        return {"centres": data["centres"].astype(np.float32), "unusual": float(data["unusual"]),
                "not_scene": float(data["not_scene"])}
    except (OSError, KeyError, ValueError):
        return None


def check_scene(features):
    """How much the photo looks like the training photos: ok, unusual, or not_scene."""
    if scene_check is None or features is None:
        return {"status": "ok", "similarity": None}
    f = features / (np.linalg.norm(features) + 1e-8)
    similarity = float((scene_check["centres"] @ f).max())
    if similarity < scene_check["not_scene"]:
        status = "not_scene"
    elif similarity < scene_check["unusual"]:
        status = "unusual"
    else:
        status = "ok"
    return {"status": status, "similarity": round(similarity, 4)}


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


scene_check = load_scene_check(SCENE_CHECK_PATH)

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

    outputs = model.run(None, {model.get_inputs()[0].name: preprocess(image)})
    probs = np.exp(outputs[0][0])  # the model ends in LogSoftmax, so exp() gives probabilities
    features = outputs[1][0] if len(outputs) > 1 else None  # older model files have no features output
    top3 = [{"label": CLASSES[i], "probability": round(float(probs[i]), 4)} for i in np.argsort(-probs)[:3]]
    best = top3[0]
    scene = check_scene(features)
    in_scope = scene["status"] != "not_scene"
    return {
        "ok": True,
        "label": best["label"],  # the model's best guess, even when in_scope is false
        "confidence": best["probability"],
        "in_scope": in_scope,
        "scene_check": scene,
        "low_confidence": best["probability"] < LOW_CONFIDENCE,
        "description": CLASS_INFO.get(best["label"], ""),
        "top3": top3,
        "comparison": compare_top_classes(top3) if in_scope else None,
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


@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    """Classifies several photos at once. One bad photo does not fail the others."""
    if model is None:
        return jsonify(ok=False, error="The model is not available on the server right now."), 503
    files = request.files.getlist("images")
    if not files:
        return jsonify(ok=False, error="Please send at least one photo."), 400
    if len(files) > MAX_BATCH:
        return jsonify(ok=False, error=f"Please send at most {MAX_BATCH} photos per request."), 400
    results = []
    for f in files:
        try:
            results.append(classify(f))
        except UserError as err:
            results.append({"ok": False, "error": str(err), "filename": f.filename})
    return jsonify(ok=True, results=results)


@app.route("/organize")
def organize():
    return render_template("organize.html", **page_context(max_batch=MAX_BATCH, max_organize=MAX_ORGANIZE))


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
