"""
Standalone inference script for MURMUR MS-TCN breath gesture classifier.
"""
import argparse
import numpy as np
import torch
import librosa

from src.train.train_ms_tcn_2c import MSTCN
from src.features.mel_delta import mel_delta_features
from src.utils.device import pick_device


def run_inference(wav_path: str, ckpt_path: str, 
                  max_len: int = 1024) -> dict:
    """
    Run inference on a single WAV file.
    
    Args:
        wav_path: Path to input WAV file
        ckpt_path: Path to trained model checkpoint (.pt)
        max_len: Fixed temporal length (must match training)
    
    Returns:
        dict with predicted class and confidence score
    """
    device = pick_device()

    # Load model
    model = MSTCN(in_ch=3, n_classes=2).to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device))
    model.eval()

    # Load and peak-normalise audio
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    m = np.max(np.abs(y))
    y = y / (m + 1e-9) if m > 0 else y

    # Extract features matching training pipeline exactly
    raw = mel_delta_features(y, sr=sr)  # [3, 64, T]

    # Apply CMVN — must match training
    mean = raw.mean(axis=(1, 2), keepdims=True)
    std  = raw.std(axis=(1, 2),  keepdims=True) + 1e-8
    raw  = (raw - mean) / std

    # Pad or crop to max_len
    T = raw.shape[-1]
    if T < max_len:
        pad = np.zeros(
            (3, 64, max_len - T), 
            dtype=raw.dtype)
        raw = np.concatenate([raw, pad], axis=-1)
    else:
        raw = raw[:, :, :max_len]

    x = torch.tensor(raw).unsqueeze(0).float().to(device)

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

    classes = ["long", "short"]
    return {
        "predicted": classes[pred.item()],
        "confidence": round(conf.item(), 4)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MURMUR inference script")
    parser.add_argument("--wav",  required=True, help="Path to input WAV file")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--max_len", type=int, default=1024,
                        help="Temporal length — must match training (default: 1024)")
    args = parser.parse_args()

    result = run_inference(args.wav, args.ckpt, args.max_len)
    print(f"Predicted: {result['predicted']}  Confidence: {result['confidence']:.4f}")
