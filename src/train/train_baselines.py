# Baseline CNN and CNN-BiLSTM models for breath gesture classification.
# Trained on the same subject-wise split and protocol as MS-TCN for fair comparison.
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Reuse dataset class and utilities from the MS-TCN training script
from src.train.train_ms_tcn_2c import MelClipSet, class_weights, evaluate
from src.utils.device import pick_device

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 7
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ─── Shared CNN Frontend ──────────────────────────────────────────────────────
def _cnn_frontend():
    """Three conv blocks: 3→32→64→128 channels, each with BN+ReLU+MaxPool(2,2)."""
    return nn.Sequential(
        # Block 1
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        # Block 2
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        # Block 3
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )


# ─── Model 1: CNN Baseline ────────────────────────────────────────────────────
class CNNBaseline(nn.Module):
    """Plain CNN: three conv blocks → global average pool → linear classifier."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.cnn = _cnn_frontend()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.cnn(x)           # [B, 128, H', T']
        h = self.pool(h)          # [B, 128, 1, 1]
        h = h.view(h.size(0), -1) # [B, 128]
        return self.classifier(h)


# ─── Model 2: CNN-BiLSTM ──────────────────────────────────────────────────────
class CNNBiLSTM(nn.Module):
    """CNN frontend + 2-layer BiLSTM over the time axis."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.cnn = _cnn_frontend()
        # Collapse frequency axis to 1, keep time dimension
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=False,
            bidirectional=True,
        )
        self.classifier = nn.Linear(128, n_classes)  # 64 * 2 directions = 128

    def forward(self, x):
        h = self.cnn(x)                  # [B, 128, H', T']
        h = self.freq_pool(h)            # [B, 128, 1, T']
        h = h.squeeze(2)                 # [B, 128, T']
        h = h.permute(2, 0, 1)          # [T', B, 128]
        _, (hn, _) = self.lstm(h)        # hn: [4, B, 64]  (2 layers × 2 dirs)
        # Concatenate forward/backward final hidden states of the last layer
        fwd = hn[-2]                     # [B, 64]
        bwd = hn[-1]                     # [B, 64]
        h = torch.cat([fwd, bwd], dim=1) # [B, 128]
        return self.classifier(h)


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_one_model(model, tr, va, te, device, classes, split_csv, ckpt_path):
    """Train model with the same protocol as MS-TCN; return (test_acc, test_f1)."""
    crit = nn.CrossEntropyLoss(
        weight=class_weights(split_csv, classes).to(device)
    )
    opt = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    best_f1, bad = -1, 0
    MAX_EPOCHS = 15
    PATIENCE = 6

    for ep in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0

        for X, y in tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(tr)
        va_acc, va_f1 = evaluate(model, va, device)
        scheduler.step(va_f1)

        print(
            f"  epoch {ep:02d} | train_loss {avg_loss:.4f} | "
            f"val_acc {va_acc:.3f} | val_f1 {va_f1:.3f}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            bad = 0
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  Early stopping at epoch {ep}.")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    te_acc, te_f1 = evaluate(model, te, device)
    return te_acc, te_f1


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(a):
    device = pick_device()
    print(f"Device: {device}\n")

    df = pd.read_csv(a.split_csv)
    classes = sorted(df[df.split == "train"].label.unique())

    trainset = MelClipSet(a.split_csv, "train", a.max_len, classes)
    valset   = MelClipSet(a.split_csv, "val",   a.max_len, classes)
    testset  = MelClipSet(a.split_csv, "test",  a.max_len, classes)

    tr = DataLoader(trainset, batch_size=a.bs, shuffle=True,  num_workers=2, pin_memory=True)
    va = DataLoader(valset,   batch_size=a.bs, shuffle=False, num_workers=2, pin_memory=True)
    te = DataLoader(testset,  batch_size=a.bs, shuffle=False, num_workers=2, pin_memory=True)

    results = {}

    # ── CNN Baseline ──────────────────────────────────────────────────────────
    print("=" * 50)
    print("Training: CNN Baseline")
    print("=" * 50)
    cnn_model = CNNBaseline(n_classes=len(classes)).to(device)
    acc, f1 = train_one_model(
        cnn_model, tr, va, te, device, classes,
        a.split_csv, a.cnn_ckpt
    )
    results["CNN Baseline"] = (acc, f1)
    print(f"→ Test acc {acc:.3f} | Test macro-F1 {f1:.3f}\n")

    # ── CNN-BiLSTM ────────────────────────────────────────────────────────────
    print("=" * 50)
    print("Training: CNN-BiLSTM")
    print("=" * 50)
    bilstm_model = CNNBiLSTM(n_classes=len(classes)).to(device)
    acc, f1 = train_one_model(
        bilstm_model, tr, va, te, device, classes,
        a.split_csv, a.bilstm_ckpt
    )
    results["CNN-BiLSTM"] = (acc, f1)
    print(f"→ Test acc {acc:.3f} | Test macro-F1 {f1:.3f}\n")

    # ── Comparison Table ──────────────────────────────────────────────────────
    print()
    print("=" * 45)
    print("BASELINE COMPARISON (subject-wise test set)")
    print("=" * 45)
    print(f"{'Model':<22} {'Acc (%)':>8}  {'Macro F1':>8}")
    print("-" * 45)
    for name, (acc, f1) in results.items():
        print(f"{name:<22} {acc * 100:>7.1f}   {f1:>8.3f}")
    print(f"{'MS-TCN (proposed)':<22} {'72.6':>7}   {'0.726':>8}")
    print("=" * 45)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train CNN and CNN-BiLSTM baselines for breath gesture classification."
    )
    ap.add_argument("--split_csv", default="manifests/split_2c_subjectwise.csv",
                    help="Path to the manifest CSV (filepath/label/split columns).")
    ap.add_argument("--max_len", type=int, default=1024,
                    help="Temporal length to pad/truncate features to.")
    ap.add_argument("--bs", type=int, default=8, help="Batch size.")
    ap.add_argument("--cnn_ckpt", default="models/cnn_baseline.pt",
                    help="Save path for the best CNN Baseline checkpoint.")
    ap.add_argument("--bilstm_ckpt", default="models/cnn_bilstm.pt",
                    help="Save path for the best CNN-BiLSTM checkpoint.")

    # Colab-friendly: parse_known_args ignores Jupyter's injected flags
    if "__file__" not in globals():
        args = ap.parse_args([])
    else:
        args, _ = ap.parse_known_args()

    main(args)