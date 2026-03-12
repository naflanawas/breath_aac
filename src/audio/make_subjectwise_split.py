import pandas as pd
import numpy as np
import os

# Reproducibility
SEED = 7
np.random.seed(SEED)

# Load segment-level metadata
seg = pd.read_csv("manifests/segments_2c.csv")

# Subject-wise split
subjects = seg["subject_id"].unique()
np.random.shuffle(subjects)

n = len(subjects)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

train_s = set(subjects[:n_train])
val_s   = set(subjects[n_train:n_train + n_val])
test_s  = set(subjects[n_train + n_val:])

def assign_split(subject_id):
    if subject_id in train_s:
        return "train"
    if subject_id in val_s:
        return "val"
    return "test"

seg["split"] = seg["subject_id"].apply(assign_split)

# Map audio segment paths to extracted feature paths
seg["filepath"] = (
    seg["filepath"]
    .str.replace("data_segments", "features/mel_dd_subjectwise", regex=False)
    .str.replace(".wav", ".npy", regex=False)
)

# Remove rows with missing feature files
before = len(seg)
seg = seg[seg["filepath"].apply(os.path.exists)].reset_index(drop=True)
after = len(seg)

print(f"Dropped {before - after} samples due to missing features")

# Save final split file
out_path = "manifests/split_2c_subjectwise.csv"
seg.to_csv(out_path, index=False)

# Summary
print("\nSaved:", out_path)
print("\nSamples per split:")
print(seg["split"].value_counts())

print("\nUnique subjects per split:")
print(seg.groupby("split")["subject_id"].nunique())
