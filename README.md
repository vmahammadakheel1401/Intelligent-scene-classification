# AI Approach for Intelligent Scene Image Classification

**Team B2 · Dept. of AIML, Dayananda Sagar College of Engineering · III semester project**

A web app that looks at a photo and says which of six scene types it shows (**buildings, forest, glacier, mountain, sea, street**), how confident it is, and how likely each of the other scenes is. It uses a ResNet50 deep-learning model fine-tuned on the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset, served by a small Flask backend.

![Classify page: photo on the left, result with all six scenes on the right](docs/screenshot.jpg)

> **Model status:** the `.pt` file in this repo must be retrained before the predictions mean anything. The original trained model was never published (see [Project history](#project-history-and-credits)). Run `train_resnet50_v2.ipynb` on Kaggle ([steps below](#retrain-the-model-on-kaggle)). The app shows a "Not sure" warning and the About page says "Results pending" until then.

---

## Features

- **Upload, drag and drop, or paste** a photo (JPG, PNG, WEBP, BMP or GIF, up to 10 MB), with a preview.
- **Results without reloading the page**: the predicted scene, the confidence, and a bar for **all six scenes** with the top 3 ranked.
- **Honest uncertainty**: a "Not sure" warning when no scene reaches 50%.
- **12 sample photos** you can click to test the model, captioned with what each photo really shows.
- **Recent results**, saved only in your browser.
- **Light and dark mode**: follows your system setting, with a toggle.
- **About the model** page: how it works, test accuracy, per-scene scores and the confusion matrix (filled in automatically after retraining), training setup, limitations and credits.
- **Never crashes on bad input**: no file, wrong file type or too-big files get a clear message instead of an error page.
- **Accessible**: works with the keyboard, results are announced to screen readers, text meets WCAG AA contrast, and it still works without JavaScript.
- **JSON API** (`POST /api/predict`) for other programs.

## Project structure

```
app.py                          Flask app: routes, validation, model loading, prediction
templates/
  base.html                     Shared layout (header, theme toggle, footer)
  index.html                    Classify page
  about.html                    About the model page
  error.html                    Friendly 404 / 500 page
static/
  css/style.css                 All styles (light and dark themes)
  js/app.js                     Upload, preview, validation, results, samples, history
  js/theme.js                   Light/dark toggle
  thumbs/                       Small versions of the sample photos
  test_image_*.jpg              12 sample photos
  favicon.svg
natural_scenes_image_resnet_classifier.pt   Model weights used by the app
model_meta.json                 Written by the v2 notebook: input size, classes, test results (after retraining)
train_resnet50_v2.ipynb         Fixed training notebook (run this on Kaggle)
natural-scenes-image-classification-using-cnns.ipynb   Original notebook (kept for reference)
requirements.txt                Python packages (pinned versions)
Dockerfile, .dockerignore       Container setup (CPU-only)
docs/                           Screenshots for this README
```

## How it works

1. **Prepare the photo.** It is resized to 224 × 224 pixels (128 × 128 for the old model) and normalised the same way as during training.
2. **Extract features.** ResNet50, pre-trained on ImageNet's 1.2 million photos, turns the image into 2,048 numbers that describe what it sees.
3. **Classify.** A small head (2048 → 512 → 128 → 6, ending in LogSoftmax) turns those numbers into a probability for each scene. The highest one is the answer.

## Run it on your computer

You need **Python 3.10–3.12**.

### Windows (PowerShell)

```powershell
git clone https://github.com/vmahammadakheel1401/Intelligent-scene-classification.git
cd Intelligent-scene-classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:8000**.

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

The first command installs the small CPU-only version of PyTorch. If you have an NVIDIA GPU, you can install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) instead; the app uses the GPU automatically.

### Options

| Environment variable | Default | What it does |
|---|---|---|
| `PORT` | `8000` | Port for `python app.py` |
| `FLASK_DEBUG` | off | Set to `1` for development error pages (never in a demo or on a server) |
| `MODEL_PATH` | `natural_scenes_image_resnet_classifier.pt` | Use a different model file |
| `MODEL_META_PATH` | `model_meta.json` | Use a different metadata file |

## Run it with Docker

```bash
docker build -t scene-classifier .
docker run -p 8000:8000 scene-classifier
```

Then open http://localhost:8000. The image uses CPU-only PyTorch and runs gunicorn with 2 workers.

## Retrain the model on Kaggle

Takes about 30–45 minutes on Kaggle's free GPU.

1. On Kaggle: **Create → New Notebook → File → Import Notebook**, and choose `train_resnet50_v2.ipynb`.
2. Right panel: **Add Input**, search **intel-image-classification** (by *puneet6060*), and add it.
3. **Session options:** Accelerator **GPU T4 x2** (or P100), Internet **On** (needed to download the ImageNet weights; requires a phone-verified Kaggle account).
4. **Run All.**
5. From the **Output** panel, download `natural_scenes_image_resnet_classifier.pt` (about 49 MB) and `model_meta.json`.
6. Put both files in the project folder (replace the old `.pt`), restart the app, and commit them.

What the v2 notebook fixes compared with the original:

| Original | v2 |
|---|---|
| Random **vertical** flips (upside-down mountains and seas) | Horizontal flips only, rotation ±10° |
| 128 px input | 224 px (the size ResNet50 was trained at) |
| Whole backbone frozen | `layer4` fine-tuned with a 10× smaller learning rate |
| Validation images were augmented | Validation uses test-time transforms |
| Loss logged from the last batch only | Averaged over every batch |
| Gradient clipping before `backward()` (no effect) | Clipping after `backward()` |
| `StepLR.step(val_loss)`, so the learning rate never dropped | Cosine schedule, stepped once per epoch |
| Last epoch saved | Best validation epoch saved |
| 99 MB pickled model | 49 MB half-precision weights, loadable with PyTorch's safe loader |

## Results

| Model | Test accuracy (3,000 photos) | Notes |
|---|---|---|
| Original custom CNN (original notebook) | 80.7% | Reported in the original notebook |
| Original ResNet50 (original notebook) | 86.1% | Reported, but the trained file was never published |
| ResNet50 v2 (this repo) | *pending: fill in after retraining* | Expected around 91–93% |

After retraining, the exact numbers, per-scene precision/recall and the confusion matrix appear on the app's **About the model** page.

## API

`POST /api/predict` with a form field called `image`:

```bash
curl -F "image=@static/test_image_3.jpg" http://127.0.0.1:8000/api/predict
```

```json
{
  "ok": true,
  "label": "mountain",
  "confidence": 0.9312,
  "low_confidence": false,
  "description": "Peaks, ridges and rocky slopes, often with snow.",
  "top3": [{"label": "mountain", "probability": 0.9312}, {"label": "glacier", "probability": 0.0521}, {"label": "sea", "probability": 0.0074}],
  "probabilities": [{"label": "buildings", "probability": 0.0011}, "... all six ..."],
  "filename": "test_image_3.jpg"
}
```

(Example values.) Errors return `{"ok": false, "error": "..."}` with status 400 (bad file), 413 (over 10 MB) or 503 (model not loaded).

## Known limitations

- The model only knows six scenes. Any other photo (a room, a person, a desert) still gets one of the six answers, which is why low-confidence results are flagged.
- Glacier vs mountain and street vs buildings are the most common mix-ups.
- Training photos are only 150 × 150 px, so fine detail in large photos is lost.

## Project history and credits

This project builds on **[sayamalt/natural-scenes-image-classification-using-cnns](https://github.com/sayamalt/natural-scenes-image-classification-using-cnns)** by Sayam Kumar. The original notebook, the model design, the sample photos and the first version of the Flask app come from that repository, which does not state a license.

What Team B2 added or changed:

- Found that the model file in the repo was untrained (classifier head at random initialisation, ~17% per class), and wrote `train_resnet50_v2.ipynb` with the training bugs fixed.
- Rewrote the backend: input validation, friendly errors instead of crashes, top-3 and all-class probabilities, a JSON API, a 10 MB upload limit, debug mode off by default, a safe weights-only model loader.
- Rebuilt the whole interface: preview, drag and drop, samples, confidence bars, uncertainty warning, history, dark mode, mobile layout, accessibility, About page.
- Cleaned the repo (removed broken Git LFS pointer files, pinned versions, smaller CPU Docker image) and wrote this README.

Dataset: [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (originally published by Intel for an Analytics Vidhya challenge).

## Team B2

| Name | USN | Role |
|---|---|---|
| *[name]* | *[USN]* | *[role]* |
| *[name]* | *[USN]* | *[role]* |
| *[name]* | *[USN]* | *[role]* |

Guide: *[faculty name]*
