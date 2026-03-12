import csv
from pathlib import Path
import argparse

def relpath(p: Path, base: Path) -> str:
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except Exception:
        return str(p.resolve())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std_root", required=True, help="Folder containing standardized WAVs (e.g., data_std/dates)")
    ap.add_argument("--mel_root", required=True, help="Folder containing Mel npy files (e.g., features/mel)")
    ap.add_argument("--out", default="manifests/files.csv", help="Output CSV path")
    ap.add_argument("--subject", default="S001", help="Default subject_id if not inferring")
    ap.add_argument("--label", default="unknown", help="Default label")
    ap.add_argument("--split", default="train", help="Default split")
    ap.add_argument("--infer_subject_from_parent", action="store_true",
                    help="If set, subject_id will be the parent folder name of each WAV")
    args = ap.parse_args()

    repo_root = Path.cwd()
    std_root = Path(args.std_root)
    mel_root = Path(args.mel_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(std_root.rglob("*.wav"))
    rows = []
    missing_mel = 0

    for w in wavs:
        stem = w.stem  # filename without extension
        # expected mel path: features/mel/<stem>.npy
        expected = mel_root / f"{stem}.npy"
        mel_path = expected if expected.exists() else None
        if mel_path is None:
            missing_mel += 1

        subject_id = w.parent.name if args.infer_subject_from_parent else args.subject

        rows.append({
            "filepath": relpath(w, repo_root),
            "subject_id": subject_id,
            "label": args.label,
            "split": args.split,
            "std_wav": relpath(w, repo_root),
            "mel_npy": relpath(mel_path, repo_root) if mel_path else ""
        })

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath","subject_id","label","split","std_wav","mel_npy"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if missing_mel:
        print(f"Note: {missing_mel} file(s) do not yet have Mel features.")

if __name__ == "__main__":
    main()
