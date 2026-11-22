# Intelligent Scene Classification

A deep learning web app that classifies natural scene images into one of six categories using a CNN (ResNet50-based classifier) and a Flask interface.

## Overview
This project predicts the scene type from an uploaded image. It is based on the **Intel Image Classification (Natural Scenes)** dataset and supports the following classes:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

## Project Structure
- `app.py` — Flask application for image upload and prediction
- `templates/index.html` — Web UI
- `static/` — Sample test images
- `natural-scenes-image-classification-using-cnns.ipynb` — Model training/fine-tuning notebook
- `natural_scenes_image_resnet_classifier.pt` — ResNet-based model checkpoint used by the app
- `requirements.txt` — Python dependencies
- `Dockerfile` — Containerized deployment setup

## How It Works
1. User uploads an image through the web interface.
2. The image is preprocessed (resize/crop/normalize) using TorchVision transforms.
3. The trained PyTorch model predicts the most likely scene class.
4. The predicted label is displayed on the page.

## Setup & Run Locally

### 1. Prerequisites
- Python 3.10+ (recommended)
- `pip`

### 2. Install dependencies
```bash
pip install -r requirements.txt
