# Handwritten Digit Recognizer

A CNN-based handwritten digit classifier trained on MNIST, with a web interface for drawing digits in real time.

## Features

- Convolutional neural network (PyTorch) achieving **99.5% accuracy** on MNIST test set
- Web interface — draw with mouse or touch screen
- Preprocessing pipeline that mimics MNIST centering for accurate real-world inference
- Data augmentation (rotation, translation, scale) for robustness

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
pip install torch torchvision flask pillow
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

## Model Architecture

```
Input (1×28×28)
  → Conv2d(1→32, 3×3) + ReLU + MaxPool
  → Conv2d(32→64, 3×3) + ReLU + MaxPool
  → Linear(3136→128) + ReLU + Dropout(0.25)
  → Linear(128→10)
```

## Background

This project is directly inspired by the pioneering work of Yann LeCun at Bell Labs (1989), who built a similar CNN — **LeNet** — to automatically read handwritten ZIP codes for the US Postal Service. MNIST itself was derived from that original postal dataset. Modern postal systems use much deeper networks capable of reading full addresses, but the core idea is identical.

## Potential Improvements

- [ ] Add batch normalization for more stable training
- [ ] Deeper architecture with residual connections (ResNet-style)
- [ ] Elastic distortion augmentation (highly effective for handwriting)
- [ ] Show top-3 predictions with confidence scores
- [ ] Extend to full word/sentence recognition with a sequence model (CRNN)
