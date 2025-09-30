# Audio standardizer: 16 kHz mono, normalized (robust to empty/bad files)
import soundfile as sf, librosa, numpy as np
from pathlib import Path
import argparse

def standardize(in_path: Path, out_path: Path, sr=16000) -> bool:
    try:
        y, _ = librosa.load(str(in_path), sr=sr, mono=True)
    except Exception as e:
        print(f"[LOAD-ERROR] {in_path}: {e}")
        return False

    if y is None or y.size == 0:
        print(f"[SKIP-EMPTY] {in_path}")
        return False

    m = np.max(np.abs(y))
    if m > 0:
        y = y / m  # safe normalize
    # if m == 0, y is flat; still write it so pipeline stays consistent

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, sr, subtype='PCM_16')
    return True

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", required=True)
    p.add_argument("--out_root", required=True)
    args = p.parse_args()
    in_root, out_root = Path(args.in_root), Path(args.out_root)

    wavs = list(in_root.rglob("*.wav"))
    ok = bad = 0
    for w in wavs:
        rel = w.relative_to(in_root)
        ok += standardize(w, (out_root/rel))
    print(f"[DONE] wrote {ok} files from {len(wavs)} inputs")
