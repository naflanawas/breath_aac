import argparse
from pathlib import Path
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from src.features.mel_delta import mel_delta_features

def main():
    """CLI entry point: extract Mel+Δ/ΔΔ features for a batch of WAV files.
 
    Accepts either a manifest CSV (``--manifest``) with a ``filepath`` column
    or a directory tree (``--in_root``).  Saves one ``.npy`` per clip under
    ``--out_root/<subject>/<label>/``.
    """
    ap = argparse.ArgumentParser()
    # Use manifest of segmented clips (recommended)
    ap.add_argument("--manifest", help="CSV with at least a 'filepath' column; optional 'label'")
    # Or fallback to scanning a directory tree (e.g., clips/)
    ap.add_argument("--in_root", help="Folder to scan for .wav if manifest not provided (expects clips/<label>/*.wav)")
    ap.add_argument("--out_root", required=True, help="Folder to save .npy features, e.g., features/mel_dd")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--fmin", type=int, default=50)
    ap.add_argument("--fmax", type=int, default=8000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    records = []
    if args.manifest:
        df = pd.read_csv(args.manifest)
        if "filepath" not in df.columns:
            raise ValueError("Manifest must have a 'filepath' column.")
        for _, r in df.iterrows():
            fp = Path(r["filepath"])
            label = r.get("label", fp.parent.name)  # infer from parent if not provided
            records.append((fp, str(label)))
    elif args.in_root:
        for w in Path(args.in_root).rglob("*.wav"):
            label = w.parent.name
            records.append((w, label))
    else:
        raise ValueError("Provide either --manifest or --in_root")

    done = skipped = failed = 0
    for w, label in tqdm(records, desc="Mel+Δ/ΔΔ (stacked)"):
        subject = w.parent.parent.name  
        out_path = out_root / subject / label / f"{w.stem}.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            y, _ = librosa.load(str(w), sr=args.sr, mono=True)
            # simple peak norm (safe for deltas)
            m = np.max(np.abs(y))
            if m > 0:
                y = y / (m + 1e-9)

            # STE-based VAD — same parameters as BreathDetector
            frame_length = 512
            hop = 256
            energy = np.array([
                np.sum(y[i:i + frame_length] ** 2)
                for i in range(0, len(y) - frame_length, hop)
            ])

            if energy.max() > 0.005:
                energy_norm = energy / energy.max()
                active = energy_norm > 0.02
                # Find longest active segment
                best_start, best_end, cur_start = 0, 0, 0
                in_seg = False
                for i, a in enumerate(active):
                    if a and not in_seg:
                        cur_start = i * hop
                        in_seg = True
                    elif not a and in_seg:
                        cur_end = i * hop
                        if (cur_end - cur_start) > (best_end - best_start):
                            best_start, best_end = cur_start, cur_end
                        in_seg = False
                if in_seg:
                    cur_end = len(y)
                    if (cur_end - cur_start) > (best_end - best_start):
                        best_start, best_end = cur_start, cur_end

                if best_end > best_start:
                    # Add 50ms padding
                    best_start = max(0, best_start - 800)
                    best_end = min(len(y), best_end + 800)
                    y = y[best_start:best_end]

            feat = mel_delta_features(y, sr=args.sr, n_fft=args.n_fft, hop=args.hop,
                                      n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, feat)
            done += 1
        except Exception as e:
            print(f"[ERROR] {w}: {e}")
            failed += 1

    print(f"Saved: {done} | Skipped: {skipped} | Failed: {failed}")
    print(f"Output: {out_root.resolve()}")

if __name__ == "__main__":
    main()
