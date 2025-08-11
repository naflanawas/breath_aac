import argparse
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm

def mel_delta_features(y, sr=16000, n_fft=1024, hop=256, n_mels=64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    d1 = librosa.feature.delta(S_db, width=9)
    d2 = librosa.feature.delta(S_db, order=2, width=9)
    feat = np.concatenate([S_db, d1, d2], axis=0).astype(np.float32)  # [3*n_mels, T]
    return feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Folder with standardized WAVs, e.g., data_std/dates")
    ap.add_argument("--out_root", required=True, help="Folder to save .npy features, e.g., features/mel")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if .npy exists")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_root.rglob("*.wav"))
    if not wavs:
        print(f"No .wav files found under {in_root}")
        return

    done, skipped, failed = 0, 0, 0
    for w in tqdm(wavs, desc="Mel+Δ/ΔΔ"):
        out_path = out_root / f"{w.stem}.npy"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            y, _ = librosa.load(str(w), sr=args.sr, mono=True)
            if np.max(np.abs(y)) > 0:
                y = y / (np.max(np.abs(y)) + 1e-9)
            feat = mel_delta_features(y, sr=args.sr, n_fft=args.n_fft, hop=args.hop, n_mels=args.n_mels)
            np.save(out_path, feat)
            done += 1
        except Exception as e:
            print(f"[ERROR] {w}: {e}")
            failed += 1

    print(f"Saved: {done} | Skipped (exists): {skipped} | Failed: {failed}")
    print(f"Output folder: {out_root.resolve()}")

if __name__ == "__main__":
    main()
