import io
import os
import base64

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from stage2_preprocessing import opencv_transform
from stage3_cnn import CIFAR10_CNN

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024   # 8 MB upload limit


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CIFAR10_CNN().to(device)
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
model.eval()
print(f"Model loaded on {device}")


def allowed_file(filename: str) -> bool:
    """Validate file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_from_pil(pil_image: Image.Image) -> list[dict]:
    """Run a PIL image through the model and return top-3 predictions."""
    tensor = opencv_transform(pil_image, augment=False)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    probs = torch.softmax(outputs, dim=1).squeeze()

    top3_probs, top3_indices = torch.topk(probs, 3)
    return [
        {"label": CLASS_NAMES[idx.item()], "confidence": round(prob.item() * 100, 1)}
        for idx, prob in zip(top3_indices, top3_probs)
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts either:
      - A file upload via multipart/form-data  (field name: "file")
      - A base64-encoded image via JSON        (field name: "image_b64")
    Returns JSON: { predictions: [{label, confidence}, ...], error: str|null }
    """
    try:
        pil_image = None

        # file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected", "predictions": []}), 400
            if not allowed_file(file.filename):
                return jsonify({"error": "File type not allowed. Use JPG, PNG, WEBP, or BMP.",
                                "predictions": []}), 400
            image_bytes = file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        elif request.is_json and "image_b64" in request.json:
            b64_data = request.json["image_b64"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        else:
            return jsonify({"error": "No image provided", "predictions": []}), 400

        predictions = predict_from_pil(pil_image)
        return jsonify({"predictions": predictions, "error": None})

    except Exception as e:
        return jsonify({"error": str(e), "predictions": []}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
