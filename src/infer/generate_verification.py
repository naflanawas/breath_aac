import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import os

SPLIT_CSV    = "/Users/nafla/Desktop/MURMUR/breath_aac/manifests/split_2c_subjectwise.csv"
COSWARA_ROOT = "/Users/nafla/Desktop/MURMUR/breath_aac/data_segments"
OUT_DIR      = os.path.expanduser("~/Documents/augmentation_verification")

df = pd.read_csv(SPLIT_CSV)
test_subjects = sorted(df[df["split"] == "test"]["subject_id"].unique())

# Find first THREE test subjects that have both WAV files
found = []
for sid in test_subjects:
    short_path = os.path.join(COSWARA_ROOT, sid, "short", "breathing-shallow.wav")
    long_path  = os.path.join(COSWARA_ROOT, sid, "long",  "breathing-deep.wav")
    if os.path.exists(short_path) and os.path.exists(long_path):
        found.append(sid)
    if len(found) == 3:
        break

print(f"Found {len(found)} subjects: {found}")

def generate_versions(wav_path, prefix, subject_out_dir):
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    m = np.max(np.abs(y))
    if m > 0:
        y = y / m

    versions = {
        "1_original":   y.copy(),
        "2_louder":     np.clip(y * 1.15, -1.0, 1.0),
        "3_quieter":    y * 0.70,
        "4_timeshift":  np.concatenate([np.zeros(int(0.05 * sr)), y])[:len(y)],
        "5_noise":      np.clip(y + np.random.normal(0, 0.01, len(y)), -1.0, 1.0),
    }

    os.makedirs(subject_out_dir, exist_ok=True)
    for name, aug_y in versions.items():
        out_path = os.path.join(subject_out_dir, f"{prefix}_{name}.wav")
        sf.write(out_path, aug_y, 16000)
        print(f"  Saved: {prefix}_{name}.wav  ({len(aug_y)/16000:.2f}s)")

for i, sid in enumerate(found, 1):
    print(f"\n--- Subject {i}: {sid} ---")
    
    # Create a subfolder for this specific subject
    subject_folder_name = f"subject{i}_{sid}"
    subject_out_dir = os.path.join(OUT_DIR, subject_folder_name)
    os.makedirs(subject_out_dir, exist_ok=True)
    
    short_wav = os.path.join(COSWARA_ROOT, sid, "short", "breathing-shallow.wav")
    long_wav  = os.path.join(COSWARA_ROOT, sid, "long",  "breathing-deep.wav")
    
    # Generate files directly into the subject's subfolder
    generate_versions(short_wav, "short", subject_out_dir)
    generate_versions(long_wav,  "long",  subject_out_dir)

print(f"\nDone. 30 files saved across 3 subject folders in: {OUT_DIR}")