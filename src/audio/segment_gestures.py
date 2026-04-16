#!/usr/bin/env python3
"""
Final segmentation script for Murmur (Coswara).

This script segments Coswara breathing recordings into
SHORT and LONG gesture classes using dataset-provided
semantic labels:

- breathing-shallow.wav -> short puff
- breathing-deep.wav    -> long puff

Rationale:
Coswara breathing recordings contain a single instructed
gesture per file. Duration-based or percentile-based
segmentation is therefore unnecessary and unreliable.

Input structure:
  Coswara-Data-master/YYYYMMDD/YYYYMMDD/<subject_id>/breathing-*.wav

Output structure:
  data_segments/<subject_id>/{short,long}/breathing-*.wav

Manifest:
  manifests/segments_2c.csv
"""

import argparse
import pathlib
import pandas as pd
import librosa
import soundfile as sf

def run(in_root, out_root, manifest_csv, sr=16000):
    """Segment Coswara breathing recordings into short/long gesture clips.
 
    Assigns class labels from the dataset filename convention:
    ``breathing-shallow.wav`` -> ``short``, ``breathing-deep.wav`` -> ``long``.
    Clips are written to ``<out_root>/<subject_id>/<label>/`` and a manifest
    CSV is saved at ``manifest_csv``.
 
    Args:
        in_root: Path to the Coswara-Data-master root directory.
        out_root: Destination directory for segmented clips.
        manifest_csv: Output path for the segments manifest CSV.
        sr: Target sample rate in Hz.
    """
    in_root = pathlib.Path(in_root)
    out_root = pathlib.Path(out_root)
    rows = []

    wavs = list(in_root.rglob("breathing-*.wav"))
    print(f"Found {len(wavs)} breathing files")

    kept = 0
    skipped = 0

    for p in wavs:
        subject_id = p.parent.name
        fname = p.name.lower()

        # Semantic label from dataset naming
        if "shallow" in fname:
            label = "short"
        elif "deep" in fname:
            label = "long"
        else:
            skipped += 1
            continue
        
        # Load audio
        try:
            y, _ = librosa.load(p.as_posix(), sr=sr, mono=True)
        except Exception as e:
            print(f"[LOAD-ERROR] {p}: {e}")
            skipped += 1
            continue

        if y is None or len(y) == 0:
            skipped += 1
            continue

        # Save clip (no further segmentation)
        out_path = out_root / subject_id / label / p.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path.as_posix(), y, sr)

        rows.append([
            str(out_path),
            label,
            subject_id,
            p.name
        ])
        kept += 1

    # Save manifest
    pd.DataFrame(
        rows,
        columns=["filepath", "label", "subject_id", "source_file"]
    ).to_csv(manifest_csv, index=False)

    print(f"[DONE] kept {kept} clips | skipped {skipped}")
    print(f"Saved -> {manifest_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Path to Coswara-Data-master")
    ap.add_argument("--out_root", required=True, help="Output directory")
    ap.add_argument("--manifest", default="manifests/segments_2c.csv")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    run(args.in_root, args.out_root, args.manifest, sr=args.sr)
