# Handwritten Digit Recognizer

A CNN-based handwritten digit classifier trained on MNIST, with a web interface for drawing digits in real time.

## Features

- Convolutional neural network (PyTorch) achieving **99.57% accuracy** on MNIST test set
- Multi-digit recognition — draw several digits at once, each recognized individually
- Auto-predict after drawing pause, undo (Ctrl+Z), brush size, light/dark canvas
- Grad-CAM heatmap overlay per digit showing which pixels influenced the prediction
- Confusion matrix at `/confusion.png`
- Data augmentation (affine + elastic distortion) for real-world robustness
- EMNIST support: `python handwriting_recognition.py --dataset emnist` (47 classes)

## Project Structure

```
digit-recognizer/
├── handwriting_recognition.py  # Model definition, training, evaluation
├── app.py                      # Flask server + /predict endpoint
├── digit_model.pth             # Trained model weights
└── templates/
    └── index.html              # Drawing canvas UI
```

## Setup

```bash
pip install torch torchvision flask pillow matplotlib
```

## Train

```bash
python handwriting_recognition.py
```

Trains for 10 epochs and saves weights to `digit_model.pth`.

## Run the web app

```bash
python app.py
```

Then open http://127.0.0.1:5000, draw a digit, and click **Predict**.

## Train

```bash
python handwriting_recognition.py              # MNIST digits (default)
python handwriting_recognition.py --dataset emnist  # EMNIST balanced (47 classes)
```

## Model Architecture

```
Input (1×28×28)
  → Conv2d(1→32,  3×3) + BatchNorm + ReLU + MaxPool  → 14×14
  → Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool  → 7×7
  → Conv2d(64→128,3×3) + BatchNorm + ReLU            → 7×7
  → Linear(6272→256) + BatchNorm + ReLU + Dropout(0.4)
  → Linear(256→10)
```

## Background

This project is directly inspired by the pioneering work of Yann LeCun at Bell Labs (1989), who built a similar CNN — **LeNet** — to automatically read handwritten ZIP codes for the US Postal Service. MNIST itself was derived from that original postal dataset. Modern postal systems use much deeper networks capable of reading full addresses, but the core idea is identical.

## Potential Improvements

- [x] Add batch normalization for more stable training
- [x] Deeper architecture (3 conv blocks, 128 channels)
- [x] Elastic distortion augmentation
- [x] Multi-digit segmentation
- [x] Grad-CAM heatmap per digit
- [x] 28×28 model input preview
- [x] Confusion matrix at /confusion.png
- [x] Auto-predict, undo, brush size, copy, light/dark toggle
- [x] EMNIST support (47 classes)
- [ ] Residual connections (ResNet-style)
- [ ] Extend to full word/sentence recognition with a sequence model (CRNN)
