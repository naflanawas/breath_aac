import argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ---------- Dataset ----------
class MelClipSet(Dataset):
    def __init__(self, split_csv, split, max_len=256, classes=None):
        df = pd.read_csv(split_csv)
        self.df = df[df["split"]==split].reset_index(drop=True)
        if classes is None:
            classes = sorted(df[df["split"]=="train"]["label"].unique())
        self.classes = classes
        self.c2i = {c:i for i,c in enumerate(self.classes)}
        self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = np.load(row["filepath"])  # [3, 64, T]
        T = x.shape[-1]
        if T < self.max_len:
            pad = np.zeros((x.shape[0], x.shape[1], self.max_len - T), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=-1)
        elif T > self.max_len:
            x = x[:, :, :self.max_len]
        y = self.c2i[row["label"]]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# ---------- Model ----------
class TCNBlock(nn.Module):
    def __init__(self, ch, k=3, dil=1):
        super().__init__()
        pad = ((k-1)//2) * dil
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, (1,k), padding=(0,pad), dilation=(1,dil)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, (1,k), padding=(0,pad), dilation=(1,dil)),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return x + self.net(x)

class MSTCN(nn.Module):
    def __init__(self, in_ch=3, base=64, n_classes=2, dilations=(1,2,4,8)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, (5,5), padding=(2,2)), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, (3,3), padding=(1,1)), nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(TCNBlock(base, k=3, dil=d), TCNBlock(base, k=3, dil=d)) for d in dilations
        ])
        self.fuse = nn.Conv2d(base*len(dilations), base, 1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(base, n_classes))
    def forward(self, x):
        h = self.stem(x)
        feats = [b(h) for b in self.branches]
        h = torch.cat(feats, dim=1)
        h = self.fuse(h)
        return self.head(h)

# ---------- Utils ----------
def pick_device():
    if torch.backends.mps.is_available(): return "mps"   # Apple Silicon
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def class_weights(split_csv, classes):
    df = pd.read_csv(split_csv)
    tr = df[df.split=="train"].label.value_counts().to_dict()
    counts = torch.tensor([tr.get(c,1) for c in classes], dtype=torch.float32)
    w = 1.0 / counts
    w = w * (len(classes) / w.sum())
    return w

def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for X,y in loader:
            X=X.to(device); y=y.to(device)
            logits = model(X)
            p = logits.argmax(1)
            ys += y.cpu().tolist(); ps += p.cpu().tolist()
    acc = accuracy_score(ys, ps)
    f1  = f1_score(ys, ps, average="macro")
    return acc, f1

# ---------- Train ----------
def main(a):
    device = pick_device()
    df = pd.read_csv(a.split_csv)
    classes = sorted(df[df.split=="train"].label.unique())
    n_classes = len(classes)

    trainset = MelClipSet(a.split_csv, "train", max_len=a.max_len, classes=classes)
    valset   = MelClipSet(a.split_csv, "val",   max_len=a.max_len, classes=classes)
    testset  = MelClipSet(a.split_csv, "test",  max_len=a.max_len, classes=classes)

    tr = DataLoader(trainset, batch_size=a.bs, shuffle=True,  num_workers=0)
    va = DataLoader(valset,   batch_size=a.bs, shuffle=False, num_workers=0)
    te = DataLoader(testset,  batch_size=a.bs, shuffle=False, num_workers=0)

    model = MSTCN(in_ch=3, n_classes=n_classes).to(device)
    w = class_weights(a.split_csv, classes).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt  = optim.Adam(model.parameters(), lr=a.lr)

    best_f1 = -1; patience = a.patience; bad=0
    for ep in range(a.epochs):
        model.train()
        for X,y in tr:
            X=X.to(device); y=y.to(device)
            opt.zero_grad(); logits=model(X); loss=crit(logits,y); loss.backward(); opt.step()
        va_acc, va_f1 = evaluate(model, va, device)
        print(f"epoch {ep:02d} | val_acc {va_acc:.3f} | val_f1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1; bad = 0
            torch.save(model.state_dict(), a.ckpt)
        else:
            bad += 1
            if bad >= patience: break

    # test
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    te_acc, te_f1 = evaluate(model, te, device)
    print(f"TEST acc {te_acc:.3f} | TEST f1 {te_f1:.3f} | saved {a.ckpt}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="manifests/split_2c.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)   # ~4.1 s @ 16k/256
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--ckpt", default="models/ms_tcn_2c.pt")
    a = ap.parse_args()
    main(a)
