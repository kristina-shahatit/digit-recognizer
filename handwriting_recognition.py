import argparse
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


# ── Model ──────────────────────────────────────────────────────────────────────

class DigitCNN(nn.Module):
    """
    3-block CNN with BatchNorm.
      num_classes=10  → MNIST digits
      num_classes=47  → EMNIST balanced (digits + letters)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),    # 28×28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 14×14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 7×7
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),                          # still 7×7
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


# ── EMNIST label map ───────────────────────────────────────────────────────────
# EMNIST balanced: 47 classes — digits, A-Z, then 11 ambiguous lowercase letters
EMNIST_LABELS = list(string.digits + string.ascii_uppercase + 'abdefghnqrt')


# ── Data ───────────────────────────────────────────────────────────────────────

def _emnist_fix(img):
    """EMNIST images are transposed vs MNIST; rotate + flip to correct."""
    return img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)


def get_dataloaders(dataset='mnist', batch_size=64):
    aug = [
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ElasticTransform(alpha=34.0, sigma=4.0),
    ]

    if dataset == 'mnist':
        norm = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        train_t = transforms.Compose(aug + norm)
        test_t  = transforms.Compose(norm)
        train_d = datasets.MNIST('./data', train=True,  download=True, transform=train_t)
        test_d  = datasets.MNIST('./data', train=False, download=True, transform=test_t)
    else:  # emnist balanced
        fix  = [transforms.Lambda(_emnist_fix)]
        norm = [transforms.ToTensor(), transforms.Normalize((0.1736,), (0.3317,))]
        train_t = transforms.Compose(fix + aug + norm)
        test_t  = transforms.Compose(fix + norm)
        train_d = datasets.EMNIST('./data', split='balanced', train=True,  download=True, transform=train_t)
        test_d  = datasets.EMNIST('./data', split='balanced', train=False, download=True, transform=test_t)

    return (DataLoader(train_d, batch_size=batch_size, shuffle=True),
            DataLoader(test_d,  batch_size=batch_size))


# ── Training ───────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            total_loss += criterion(out, labels).item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ── Inference on a custom image ────────────────────────────────────────────────

def predict_image(model, image_path, device):
    t = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tensor = t(Image.open(image_path)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    digit = probs.argmax().item()
    print(f"Predicted: {digit}  ({probs[digit]:.1%})")
    return digit, float(probs[digit])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'emnist'], default='mnist',
                        help='mnist (10 classes) or emnist balanced (47 classes)')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    num_classes = 10 if args.dataset == 'mnist' else 47
    save_path   = 'digit_model.pth' if args.dataset == 'mnist' else 'emnist_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dataset: {args.dataset}  |  Classes: {num_classes}  |  Device: {device}")

    train_loader, test_loader = get_dataloaders(args.dataset)
    model     = DigitCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train loss {tr_loss:.4f}  train acc {tr_acc:.2%}  "
            f"test loss {te_loss:.4f}  test acc {te_acc:.2%}"
        )

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
