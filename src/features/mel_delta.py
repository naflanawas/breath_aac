import numpy as np, librosa, argparse
from pathlib import Path

def mel_delta_features(y, sr=16000, n_fft=1024, hop=256, n_mels=64, fmin=50, fmax=8000):
    """Compute stacked log-Mel / delta / delta-delta features from a waveform.
 
    Args:
        y: 1-D float32 waveform array (peak-normalised to ±1).
        sr: Sample rate in Hz.
        n_fft: FFT window size.
        hop: Hop length in samples.
        n_mels: Number of Mel filter banks.
        fmin: Lowest frequency for the Mel filterbank.
        fmax: Highest frequency for the Mel filterbank.
 
    Returns:
        numpy array of shape (3, n_mels, T) — channels are
        [log-Mel, Δ, ΔΔ], dtype float32.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    d1   = librosa.feature.delta(S_db, width=9).astype(np.float32)
    d2   = librosa.feature.delta(S_db, order=2, width=9).astype(np.float32)
    feat = np.stack([S_db, d1, d2], axis=0)  # [3, n_mels, T]
    return feat

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    y, sr = librosa.load(args.wav, sr=16000, mono=True)
    m = np.max(np.abs(y))
    if m > 0: y = y / (m + 1e-9)
    feat = mel_delta_features(y, sr=sr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, feat)
    print("Saved:", args.out, "shape:", feat.shape)  # format (3, 64, T)
