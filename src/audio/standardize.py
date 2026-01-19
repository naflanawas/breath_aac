# Audio standardizer: fixed length, 16 kHz mono, normalized
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
import argparse

SR = 16000
SHORT_SAMPLES = int(0.8 * SR)   # 800 ms
LONG_SAMPLES  = int(2.5 * SR)   # 2500 ms

def pad_or_crop(y, target_len):
    cur = len(y)

    if cur == target_len:
        return y

    if cur > target_len:
        start = (cur - target_len) // 2
        return y[start:start + target_len]

    # pad
    pad_total = target_len - cur
    left = pad_total // 2
    right = pad_total - left
    return np.pad(y, (left, right), mode="constant")

def standardize(in_path: Path, out_path: Path) -> bool:
    try:
        y, _ = librosa.load(str(in_path), sr=SR, mono=True)
    except Exception as e:
        print(f"[LOAD-ERROR] {in_path}: {e}")
        return False

    if y is None or y.size == 0:
        print(f"[SKIP-EMPTY] {in_path}")
        return False

    # amplitude normalization
    m = np.max(np.abs(y))
    if m > 0:
        y = y / m

    # decide target length from folder name
    label = in_path.parent.name.lower()
    if label == "short":
        y = pad_or_crop(y, SHORT_SAMPLES)
    elif label == "long":
        y = pad_or_crop(y, LONG_SAMPLES)
    else:
        print(f"[SKIP-UNKNOWN] {in_path}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, SR, subtype="PCM_16")
    return True

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", required=True)
    p.add_argument("--out_root", required=True)
    args = p.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    wavs = list(in_root.rglob("*.wav"))
    ok = 0

    for w in wavs:
        rel = w.relative_to(in_root)
        out = out_root / rel
        if standardize(w, out):
            ok += 1

    print(f"[DONE] standardized {ok} / {len(wavs)} files")
