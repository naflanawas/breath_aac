import argparse
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.utils.device import pick_device

# reproducibility 
SEED = 7
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#  DATASET 
class MelClipSet(Dataset):
    """PyTorch Dataset for pre-extracted Mel+Δ/ΔΔ feature clips."""
    def __init__(self, split_csv, split, max_len=1024, classes=None):
        """Args:
            split_csv: Path to the manifest CSV with columns filepath/label/split.
            split: Which partition to load ('train', 'val', or 'test').
            max_len: Fixed temporal length in frames (pad or truncate).
            classes: Ordered list of class names; inferred from train rows if None.
        """
        df = pd.read_csv(split_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if classes is None:
            classes = sorted(df[df["split"] == "train"]["label"].unique())

        self.classes = classes
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        self.max_len = max_len

    def __len__(self):
        """Return the number of clips in this split."""
        return len(self.df)

    def __getitem__(self, i):
        """Return (feature_tensor, label_index) for clip i."""
        row = self.df.iloc[i]
        x = np.load(row["filepath"])  # [3, 64, T]

        # Pad / Crop
        T = x.shape[-1]
        if T < self.max_len:
            pad = np.zeros((x.shape[0], x.shape[1], self.max_len - T), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=-1)
        elif T > self.max_len:
            x = x[:, :, :self.max_len]

        # SpecAugment (TRAIN ONLY) 
        if self.df.iloc[i]["split"] == "train":
            # frequency mask
            f = np.random.randint(0, 8)
            f0 = np.random.randint(0, max(1, x.shape[1] - f))
            x[:, f0:f0+f, :] = 0

            # time mask
            t = np.random.randint(0, 80)
            t0 = np.random.randint(0, max(1, x.shape[2] - t))
            x[:, :, t0:t0+t] = 0

        y = self.c2i[row["label"]]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# MODEL
class TCNBlock(nn.Module):
    """Residual dilated temporal convolution block.
    Applies two 1×k Conv2d layers (dilation along the time axis) and adds
    the input as a residual skip connection.
    """
    def __init__(self, ch, k=3, dil=1):
        """Args:
            ch: Number of input/output channels.
            k: Temporal kernel size.
            dil: Temporal dilation factor.
        """
        super().__init__()
        pad = ((k - 1) // 2) * dil
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, (1, k), padding=(0, pad), dilation=(1, dil)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, (1, k), padding=(0, pad), dilation=(1, dil)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply residual TCN block: output = x + conv(x)."""
        return x + self.net(x)

class MSTCN(nn.Module):
    """Multi-Scale Temporal Convolutional Network for breath gesture classification.
 
    Architecture:
        1. Stem — two Conv2d layers for local 2-D feature extraction.
        2. Branches — one TCNBlock pair per dilation value, run in parallel.
        3. Fuse — 1×1 Conv2d to merge all branch outputs.
        4. Pool — AdaptiveAvgPool2d to (1,1).
        5. Embed — linear projection.
        6. Classifier — linear head (omitted when return_embedding=True).
    """
    def __init__(self, in_ch=3, base=64, n_classes=2, dilations=(1, 2, 4, 8)):
        """Args:
            in_ch: Number of input channels (3 for log-Mel+Δ+ΔΔ).
            base: Base channel width.
            n_classes: Number of output classes.
            dilations: Tuple of temporal dilation factors for the parallel branches.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, (5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                TCNBlock(base, k=3, dil=d),
                TCNBlock(base, k=3, dil=d)
            ) for d in dilations
        ])

        self.fuse = nn.Conv2d(base * len(dilations), base, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = nn.Linear(base, base)
        self.classifier = nn.Linear(base, n_classes)


    def forward(self, x, return_embedding=False):
        """Forward pass."""
        h = self.stem(x)
        feats = [b(h) for b in self.branches]
        h = torch.cat(feats, dim=1)
        h = self.fuse(h)
        h = self.pool(h)
        h = h.view(h.size(0), -1)   # [B, 64]

        if return_embedding:
            return h

        h = self.embed(h)
        return self.classifier(h)

def class_weights(split_csv, classes):
    """Compute inverse-frequency class weights for weighted CrossEntropyLoss.
 
    Args:
        split_csv: Manifest CSV with label/split columns.
        classes: Ordered list of class names.
 
    Returns:
        Float32 tensor of shape (len(classes),) normalised so weights sum to
        len(classes).
    """
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
            logits = model(X)
            ps.extend(logits.argmax(1).cpu().tolist())
            ys.extend(y.cpu().tolist())

    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


# TRAIN
def main(a):
    """Train the MS-TCN model."""
    # metric logging
    train_losses = []
    val_accs = []
    val_f1s = []

    device = pick_device()

    df = pd.read_csv(a.split_csv)
    classes = sorted(df[df.split == "train"].label.unique())

    trainset = MelClipSet(a.split_csv, "train", a.max_len, classes)
    valset   = MelClipSet(a.split_csv, "val",   a.max_len, classes)
    testset  = MelClipSet(a.split_csv, "test",  a.max_len, classes)

    tr = DataLoader(trainset, batch_size=a.bs, shuffle=True, num_workers=2)
    va = DataLoader(valset,   batch_size=a.bs, num_workers=2)
    te = DataLoader(testset,  batch_size=a.bs, num_workers=2)

    model = MSTCN(n_classes=len(classes)).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(a.split_csv, classes).to(device))
    opt  = optim.Adam(model.parameters(), lr=a.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=2,min_lr=1e-6)

    best_f1, bad = -1, 0
    for ep in range(a.epochs):
        model.train()
        running_loss = 0.0

        for X, y in tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(tr)
        train_losses.append(avg_train_loss)


        va_acc, va_f1 = evaluate(model, va, device)
        val_accs.append(va_acc)
        val_f1s.append(va_f1)

        scheduler.step(va_f1)

        print(
            f"epoch {ep:02d} | "
            f"train_loss {avg_train_loss:.4f} | "
            f"val_acc {va_acc:.3f} | "
            f"val_f1 {va_f1:.3f}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            bad = 0
            torch.save(model.state_dict(), a.ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                break

    # ---------------- SAVE TRAINING CURVES ----------------
    os.makedirs("logs", exist_ok=True)
    np.save("logs/train_loss.npy", np.array(train_losses))
    np.save("logs/val_acc.npy", np.array(val_accs))
    np.save("logs/val_f1.npy", np.array(val_f1s))

    model.load_state_dict(torch.load(a.ckpt))
    te_acc, te_f1 = evaluate(model, te, device)
    print(f"TEST acc {te_acc:.3f} | TEST f1 {te_f1:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--ckpt", default="models/ms_tcn_no_cmvn.pt")

    # Check if running in an interactive notebook environment (e.g., Colab)
    # In such an environment, sys.argv might not contain expected command-line arguments.
    # We provide explicit default arguments for interactive execution.
    if '__file__' not in globals(): # Heuristic for notebook environment
        args = ap.parse_args([
            "--split_csv", "manifests/split_2c_subjectwise.csv",
            "--max_len", "1024",
            "--bs", "8",
            "--ckpt", "models/ms_tcn_no_cmvn.pt"
        ])
    else:
        args = ap.parse_args() # For command-line execution, use sys.argv

    main(args)
