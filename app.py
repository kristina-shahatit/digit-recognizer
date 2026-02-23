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
    This prevents the digit from being tiny/off-center in the 28x28 input.
    """
    arr = np.array(img)
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)

    if not rows.any():
        # Canvas is empty — return blank image
        return img.resize((28, 28))

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Pad to square with 20% margin (matches MNIST centering)
    w, h = img.size
    pad = max(w, h) // 4
    img = ImageOps.expand(img, border=pad, fill=0)
    img = img.resize((28, 28), Image.LANCZOS)
    return img


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

    img = crop_and_center(img)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
        digit = probs.argmax().item()
        confidence = probs[digit].item()

    return jsonify({"digit": digit, "confidence": round(confidence * 100, 1)})


if __name__ == "__main__":
    app.run(debug=True)
