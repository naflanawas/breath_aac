#Standardize audio files to a common sample rate and format
#Audio standardizer (turn any WAV into 16k mono, normalized)

import soundfile as sf, librosa, numpy as np
from pathlib import Path
import argparse

def standardize(in_path: Path, out_path: Path, sr=16000):
    y, _ = librosa.load(str(in_path), sr=sr, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, sr, subtype='PCM_16')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", required=True)
    p.add_argument("--out_root", required=True)
    args = p.parse_args()
    in_root, out_root = Path(args.in_root), Path(args.out_root)
    wavs = list(in_root.rglob("*.wav"))
    for w in wavs:
        rel = w.relative_to(in_root)
        standardize(w, (out_root/rel))
