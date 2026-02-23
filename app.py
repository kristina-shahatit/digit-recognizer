import base64
import io
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template

# Resolve paths relative to this file so the app works from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


# ── Model (must match handwriting_recognition.py) ─────────────────────────────

class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Load model ────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitCNN().to(device)
model.load_state_dict(torch.load(os.path.join(_HERE, 'digit_model.pth'), map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping on the last Conv2d layer."""

    def __init__(self, model):
        self._acts  = None
        self._grads = None
        last_conv = None
        for m in model.features:
            if isinstance(m, nn.Conv2d):
                last_conv = m
        last_conv.register_forward_hook(
            lambda m, i, o: setattr(self, '_acts', o.detach()))
        last_conv.register_full_backward_hook(
            lambda m, gi, go: setattr(self, '_grads', go[0].detach()))

    def __call__(self, tensor: torch.Tensor, class_idx: int,
                 original: Image.Image) -> Image.Image:
        import matplotlib.cm as cm_mod

        t = tensor.clone().requires_grad_(True)
        model.eval()
        model.zero_grad()
        model(t)[0, class_idx].backward()

        weights = self._grads.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self._acts).sum(1)).squeeze().numpy()

        if cam.max() > 0:
            cam /= cam.max()

        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((28, 28), Image.LANCZOS)
        colored = (cm_mod.jet(np.array(cam_pil) / 255.0)[:, :, :3] * 255).astype(np.uint8)
        blended = Image.blend(original.convert('RGB'), Image.fromarray(colored), alpha=0.55)
        model.zero_grad()
        return blended


grad_cam = GradCAM(model)


# ── Helpers ───────────────────────────────────────────────────────────────────

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def crop_and_center(img: Image.Image) -> Image.Image:
    arr  = np.array(img)
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)
    if not rows.any():
        return img.resize((28, 28))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    img = img.crop((cmin, rmin, cmax + 1, rmax + 1))
    pad = max(*img.size) // 4
    img = ImageOps.expand(img, border=pad, fill=0)
    return img.resize((28, 28), Image.LANCZOS)


def segment_digits(img: Image.Image) -> list:
    arr         = np.array(img)
    col_has_ink = np.any(arr > 10, axis=0)
    segments, in_seg, start = [], False, 0
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
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg[0] - merged[-1][1] <= 8:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    crops = []
    for x0, x1 in merged:
        col = arr[:, x0:x1]
        rows = np.any(col > 10, axis=1)
        if not rows.any():
            continue
        y0 = int(np.where(rows)[0][0])
        y1 = int(np.where(rows)[0][-1]) + 1
        crops.append(img.crop((x0, y0, x1, y1)))
    return crops


def classify(crop: Image.Image) -> dict:
    processed = crop_and_center(crop)
    tensor    = transform(processed).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    digit      = int(probs.argmax())
    confidence = round(float(probs[digit]) * 100, 1)

    heatmap = grad_cam(tensor, digit, processed)

    return {
        'digit':      digit,
        'confidence': confidence,
        'uncertain':  confidence < 60,
        'preview':    img_to_b64(processed),
        'heatmap':    img_to_b64(heatmap),
    }


# ── Confusion matrix (generated lazily on first request) ─────────────────────

_confusion_png = None


def build_confusion_matrix():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    norm   = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    loader = DataLoader(datasets.MNIST(os.path.join(_HERE, 'data'), train=False, download=True, transform=norm), batch_size=256)

    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            preds.extend(model(imgs.to(device)).argmax(1).cpu().tolist())
            labels.extend(lbls.tolist())

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix — MNIST test set', fontsize=14)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=9,
                    color='white' if cm[i][j] > cm.max() * 0.5 else 'black')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data       = request.get_json()
    img_bytes  = base64.b64decode(data['image'].split(',')[1])
    img        = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
    background = Image.new('RGBA', img.size, (0, 0, 0, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert('L')

    crops = segment_digits(img)
    if not crops:
        return jsonify({'digits': []})
    return jsonify({'digits': [classify(c) for c in crops]})


@app.route('/confusion.png')
def confusion():
    global _confusion_png
    if _confusion_png is None:
        _confusion_png = build_confusion_matrix()
    return app.response_class(_confusion_png, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
