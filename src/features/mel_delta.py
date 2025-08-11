#Mel + Δ/ΔΔ feature extractor for a single file (quick check)


import numpy as np, librosa, argparse
from pathlib import Path

def mel_delta_features(y, sr=16000, n_fft=1024, hop=256, n_mels=64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    d1 = librosa.feature.delta(S_db, width=9)
    d2 = librosa.feature.delta(S_db, order=2, width=9)
    feat = np.concatenate([S_db, d1, d2], axis=0).astype(np.float32)  # [3*n_mels, T]
    return feat

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    y, sr = librosa.load(args.wav, sr=16000, mono=True)
    feat = mel_delta_features(y, sr=sr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, feat)
    print("Saved:", args.out, "shape:", feat.shape)
