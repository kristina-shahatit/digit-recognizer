import base64
import io
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


# ── Model definition (must match handwriting_recognition.py) ──────────────────

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Load model ────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN().to(device)
model.load_state_dict(torch.load("digit_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def crop_and_center(img: Image.Image) -> Image.Image:
    """
    Mimic MNIST preprocessing: crop tightly around the drawn pixels,
    pad to square, then resize to 28x28.
    """
    arr = np.array(img)
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)

    if not rows.any():
        return img.resize((28, 28))

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    w, h = img.size
    pad = max(w, h) // 4
    img = ImageOps.expand(img, border=pad, fill=0)
    img = img.resize((28, 28), Image.LANCZOS)
    return img


def segment_digits(img: Image.Image) -> list[Image.Image]:
    """
    Split a canvas image into individual digit crops by finding vertical gaps.
    Uses column projection: columns with no ink mark boundaries between digits.
    Intra-digit gaps ≤ 8px are merged so strokes within one digit stay together.
    Returns crops sorted left-to-right.
    """
    arr = np.array(img)
    col_has_ink = np.any(arr > 10, axis=0)  # True for each column that has ink

    # Find contiguous runs of inked columns
    segments = []
    in_seg = False
    start = 0
    for i, has_ink in enumerate(col_has_ink):
        if has_ink and not in_seg:
            in_seg, start = True, i
        elif not has_ink and in_seg:
            in_seg = False
            segments.append([start, i])
    if in_seg:
        segments.append([start, len(col_has_ink)])

    if not segments:
        return []

    # Merge segments whose gap is ≤ 8px (handles lifted strokes within one digit)
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg[0] - merged[-1][1] <= 8:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    # Crop each segment to its tight bounding box
    crops = []
    for x_start, x_end in merged:
        col_slice = arr[:, x_start:x_end]
        rows_with_ink = np.any(col_slice > 10, axis=1)
        if not rows_with_ink.any():
            continue
        y_start = int(np.where(rows_with_ink)[0][0])
        y_end   = int(np.where(rows_with_ink)[0][-1]) + 1
        crops.append(img.crop((x_start, y_start, x_end, y_end)))

    return crops


def classify(crop: Image.Image) -> tuple[int, float]:
    """Run a single digit crop through the model."""
    tensor = transform(crop_and_center(crop)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    digit = int(probs.argmax())
    return digit, round(float(probs[digit]) * 100, 1)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Strip the data:image/png;base64, prefix
    img_bytes = base64.b64decode(data["image"].split(",")[1])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Flatten alpha onto black background, then convert to grayscale
    background = Image.new("RGBA", img.size, (0, 0, 0, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("L")  # grayscale: white digit on black bg

    crops = segment_digits(img)
    if not crops:
        return jsonify({"digits": []})

    digits = [{"digit": d, "confidence": c} for d, c in (classify(crop) for crop in crops)]
    return jsonify({"digits": digits})


if __name__ == "__main__":
    app.run(debug=True)
