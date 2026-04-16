"""
Ablation study script for MURMUR.
Trains the MS-TCN with one component removed at a time.

Usage:
  python -m src.train.train_ablation \
    --split_csv manifests/split_2c_subjectwise.csv \
    --ablation none          # full system (baseline)

  python -m src.train.train_ablation \
    --split_csv manifests/split_2c_subjectwise.csv \
    --ablation no_delta      # log-Mel only, no delta features

  python -m src.train.train_ablation \
    --split_csv manifests/split_2c_subjectwise.csv \
    --ablation no_augment    # no SpecAugment

  python -m src.train.train_ablation \
    --split_csv manifests/split_2c_subjectwise.csv \
    --ablation no_cmvn       # no CMVN normalisation

  python -m src.train.train_ablation \
    --split_csv manifests/split_2c_subjectwise.csv \
    --ablation single_scale  # single dilation d=1 only
"""

import argparse, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.train.train_ms_tcn_2c import TCNBlock, MSTCN
from src.utils.device import pick_device

#  Reproducibility 
SEED = 7
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


#  Dataset 

class AblationDataset(Dataset):
    """Dataset for ablation experiments - selectively disables CMVN, SpecAugment,
    delta channels, or multi-scale dilation depending on the ``ablation`` flag."""
    def __init__(self, split_csv, split, max_len=1024, classes=None,
                 ablation="none"):
        """Args:
            split_csv: Manifest CSV with filepath/label/split columns.
            split: Partition to load ('train', 'val', or 'test').
            max_len: Fixed temporal length in frames.
            classes: Class list; inferred from train rows if None.
            ablation: One of 'none', 'no_delta', 'no_augment',
                      'no_cmvn', 'single_scale'.
        """
        df = pd.read_csv(split_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if classes is None:
            classes = sorted(df[df["split"] == "train"]["label"].unique())
        self.classes  = classes
        self.c2i      = {c: i for i, c in enumerate(classes)}
        self.max_len  = max_len
        self.ablation = ablation   # "none"|"no_delta"|"no_augment"|"no_cmvn"|"single_scale"
        self.split    = split

    def __len__(self):
        """Return number of samples in this partition."""
        return len(self.df)

    def __getitem__(self, idx):
        """Return (feature_tensor, label_index) applying ablation transforms."""
        row = self.df.iloc[idx]
        x   = np.load(row["filepath"])   # [3, 64, T]

        #  Ablation: no_delta -> keep only channel 0 (log-Mel), zero others
        if self.ablation == "no_delta":
            x = x[0:1, :, :]   # shape becomes [1, 64, T]

        #  Ablation: no_cmvn -> skip normalisation; otherwise apply CMVN
        if self.ablation != "no_cmvn":
            mean = x.mean(axis=(1, 2), keepdims=True)
            std  = x.std(axis=(1, 2),  keepdims=True) + 1e-8
            x    = (x - mean) / std

        #  Pad / crop to max_len
        T = x.shape[-1]
        if T < self.max_len:
            pad = np.zeros((x.shape[0], x.shape[1], self.max_len - T),
                           dtype=x.dtype)
            x = np.concatenate([x, pad], axis=-1)
        else:
            x = x[:, :, :self.max_len]

        #  Ablation: no_augment -> skip SpecAugment
        if self.split == "train" and self.ablation not in ("no_augment",):
            # frequency mask
            f  = np.random.randint(0, 8)
            f0 = np.random.randint(0, max(1, x.shape[1] - f))
            x[:, f0:f0 + f, :] = 0
            # time mask
            t  = np.random.randint(0, 80)
            t0 = np.random.randint(0, max(1, x.shape[2] - t))
            x[:, :, t0:t0 + t] = 0

        y = self.c2i[row["label"]]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def class_weights(split_csv, classes):
    """Compute inverse-frequency class weights (see train_ms_tcn_2c.class_weights)."""
    df = pd.read_csv(split_csv)
    tr = df[df.split == "train"].label.value_counts().to_dict()
    counts = torch.tensor([tr.get(c, 1) for c in classes], dtype=torch.float32)
    w = 1.0 / counts
    return w * (len(classes) / w.sum())


def evaluate(model, loader, device):
    """Evaluate model on a DataLoader; return (accuracy, macro-F1)."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            ps.extend(model(X).argmax(1).cpu().tolist())
            ys.extend(y.cpu().tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


#  Train 

def main(a):
    """Run one ablation condition end-to-end (train -> val -> test).
 
    Args:
        a: Parsed argparse namespace with split_csv, ablation, epochs, bs,
           max_len, lr, patience fields.
    """
    device  = pick_device()
    df      = pd.read_csv(a.split_csv)
    classes = sorted(df[df.split == "train"].label.unique())

    # For single_scale ablation, use only dilation=1
    dilations = (1,) if a.ablation == "single_scale" else (1, 2, 4, 8)

    # For no_delta, model receives 3 channels but channels 1 & 2 are zeroed
    # so in_ch stays 3 (same architecture, different input content)
    in_ch = 1 if a.ablation == "no_delta" else 3

    print(f"\n{'='*55}")
    print(f"ABLATION: {a.ablation.upper()}")
    print(f"Dilations: {dilations} | in_ch: {in_ch}")
    print(f"{'='*55}\n")

    trainset = AblationDataset(a.split_csv, "train",  a.max_len, classes, a.ablation)
    valset   = AblationDataset(a.split_csv, "val",    a.max_len, classes, a.ablation)
    testset  = AblationDataset(a.split_csv, "test",   a.max_len, classes, a.ablation)

    tr = DataLoader(trainset, batch_size=a.bs, shuffle=True,  num_workers=2)
    va = DataLoader(valset,   batch_size=a.bs, num_workers=2)
    te = DataLoader(testset,  batch_size=a.bs, num_workers=2)

    model = MSTCN(in_ch=in_ch, n_classes=len(classes), dilations=dilations).to(device)
    crit  = nn.CrossEntropyLoss(weight=class_weights(a.split_csv, classes).to(device))
    opt   = optim.Adam(model.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau( opt, mode="max", factor=0.5, patience=2, min_lr=1e-6)

    best_f1, bad = -1, 0
    ckpt = f"models/ablation_{a.ablation}.pt"
    os.makedirs("models", exist_ok=True)

    for ep in range(a.epochs):
        model.train()
        for X, y in tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()

        va_acc, va_f1 = evaluate(model, va, device)
        sched.step(va_f1)
        print(f"epoch {ep:02d} | val_acc {va_acc:.3f} | val_f1 {va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            bad = 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping at epoch {ep}")
                break

    model.load_state_dict(torch.load(ckpt))
    te_acc, te_f1 = evaluate(model, te, device)
    print(f"\n{'='*55}")
    print(f"ABLATION: {a.ablation.upper()}")
    print(f"TEST acc={te_acc:.3f} | TEST F1={te_f1:.3f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="manifests/split_2c_subjectwise.csv")
    ap.add_argument("--ablation",  default="none",
                    choices=["none","no_delta","no_augment",
                             "no_cmvn","single_scale"])
    ap.add_argument("--epochs",   type=int,   default=40)
    ap.add_argument("--bs",       type=int,   default=8)
    ap.add_argument("--max_len",  type=int,   default=1024)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--patience", type=int,   default=6)
    args = ap.parse_args()
    main(args)