"""
Standalone inference script for MURMUR MS-TCN breath gesture classifier.
"""
import argparse
import numpy as np
import torch

from src.train.train_ms_tcn_2c import MSTCN
from src.features.mel_delta import mel_delta_features
from src.utils.device import pick_device


def run_inference(wav_path: str, ckpt_path: str, max_len: int = 1024) -> dict:
    """
    Run inference on a single WAV file.
    
    Args:
        wav_path: Path to input WAV file
        ckpt_path: Path to trained model checkpoint (.pt)
        max_len: Fixed temporal length (must match training — default 1024)
    
    Returns:
        dict with predicted class and confidence score
    """
    device = pick_device()

    # Load model
    model = MSTCN(in_ch=3, num_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Extract features
    features = mel_delta_features(wav_path, max_len=max_len)
    x = torch.tensor(features).unsqueeze(0).float().to(device)

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

    classes = ["long_puff", "short_puff"]
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
