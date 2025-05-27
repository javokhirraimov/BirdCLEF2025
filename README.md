# BirdCLEF+ 2025 Kaggle Competition
Kaggle Competition: Species identification from audio, focused on birds, amphibians, mammals and insects from the Middle Magdalena Valley of Colombia.

# BirdCLEF2025 Model

This repository hosts a PyTorch model trained for the BirdCLEF 2025 challenge, classifying bird species from audio spectrograms.

---

## Model Details

- **Architecture:** EfficientNetV2-S backbone (modified for single-channel mel-spectrogram input)
- **Input:** Log-mel spectrogram of audio clips (5 seconds, 32kHz, 128 mel bins)
- **Classes:** 206 bird species
- **Loss:** Improved Focal Loss with label smoothing
- **Data Augmentation:** SpecAugment (frequency and time masking)
- **Training:** Mixed precision and early stopping applied

---

## Usage

### 1. Install Dependencies

```bash
pip install torch torchaudio timm librosa numpy

import librosa
import numpy as np
import torch

# Audio preprocessing parameters
SAMPLE_RATE = 32000
DURATION = 5  # seconds
N_MELS = 128
HOP_LENGTH = 512

def extract_logmel(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    target_length = SAMPLE_RATE * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return torch.tensor(log_mel_spec).unsqueeze(0).float()  # shape: [1, mel_bins, time_frames]

# Load your model definition here
from model import BirdCLEFModel  # ensure model.py is present

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdCLEFModel(num_classes=206, backbone='efficientnet').to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

def predict(audio_path):
    spec = extract_logmel(audio_path).unsqueeze(0).to(device)  # add batch dimension: [1, 1, 128, T]
    with torch.no_grad():
        outputs = model(spec)
        probs = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    return pred_label, probs.cpu().numpy()

# Example usage:
audio_file = "path/to/sample.ogg"
label_idx, probabilities = predict(audio_file)
print(f"Predicted label index: {label_idx}")

---


