import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


# ── Model ──────────────────────────────────────────────────────────────────────

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 → 7x7
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


# ── Data ───────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size=64):
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,           # ±10° rotation
            translate=(0.1, 0.1), # ±10% shift
            scale=(0.9, 1.1),     # ±10% zoom
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST("./data", train=True,  download=True, transform=train_transform)
    test_data  = datasets.MNIST("./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size)
    return train_loader, test_loader


# ── Training ───────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ── Inference on a custom image ────────────────────────────────────────────────

def predict_image(model, image_path, device):
    """
    Predict the digit in an image file.
    Expects a grayscale image (or converts automatically).
    White digit on black background works best; if yours is inverted,
    set invert=True below.
    """
    invert = False  # set True if your image has a black digit on white background

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    img = Image.open(image_path)
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 1, 28, 28)
    if invert:
        tensor = 1 - tensor

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        digit = probs.argmax().item()
        confidence = probs[digit].item()
    print(f"Predicted digit: {digit}  (confidence: {confidence:.1%})")
    return digit, confidence


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=64)

    model     = DigitCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    epochs = 10
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train loss {tr_loss:.4f}  train acc {tr_acc:.2%}  "
            f"test loss {te_loss:.4f}  test acc {te_acc:.2%}"
        )

    torch.save(model.state_dict(), "digit_model.pth")
    print("Model saved to digit_model.pth")

    # ── To load and run on your own image later: ──
    # model.load_state_dict(torch.load("digit_model.pth", map_location=device))
    # predict_image(model, "my_digit.png", device)


if __name__ == "__main__":
    main()
