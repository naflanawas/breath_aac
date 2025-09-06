import argparse, numpy as np, pandas as pd, torch, torch.nn as nn, librosa

# ---- Model (same as training) ----
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

# ---- Features (same settings as training) ----
def mel_delta_stack(y, sr=16000, n_mels=64, n_fft=1024, hop=256, fmin=50, fmax=8000):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                       hop_length=hop, fmin=fmin, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    d1   = librosa.feature.delta(S_db, width=9).astype(np.float32)
    d2   = librosa.feature.delta(S_db, order=2, width=9).astype(np.float32)
    return np.stack([S_db, d1, d2], axis=0)  # [3, 64, T]

def center_or_max_window(x, target_T=256):
    # x: [3, 64, T]
    T = x.shape[-1]
    if T <= target_T:
        # pad on the right
        pad = np.zeros((x.shape[0], x.shape[1], target_T - T), dtype=x.dtype)
        return np.concatenate([x, pad], axis=-1)
    # choose the most energetic window of length target_T
    # energy proxy: sum of log-mel over channels & freq
    energy = x[0].sum(axis=0)  # use log-mel channel
    win = target_T
    cs = np.cumsum(np.pad(energy, (1,0)))  # prefix sum
    best_s = max(range(0, T - win + 1), key=lambda s: cs[s+win] - cs[s])
    return x[:, :, best_s:best_s+win]

def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", default="models/ms_tcn_2c.pt")
    ap.add_argument("--split_csv", default="manifests/split_2c.csv")  # to get class order
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    # class order from training split (sorted)
    df = pd.read_csv(args.split_csv)
    classes = sorted(df[df.split=="train"]["label"].unique())

    # load model
    device = pick_device()
    model = MSTCN(in_ch=3, n_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # load audio -> features
    y, sr = librosa.load(args.wav, sr=args.sr, mono=True)
    # simple peak norm
    m = np.max(np.abs(y));  y = y/(m+1e-9) if m>0 else y
    X = mel_delta_stack(y, sr=sr)          # [3, 64, T]
    X = center_or_max_window(X, target_T=args.max_len)
    X = torch.from_numpy(X).unsqueeze(0).to(device)   # [1,3,64,256]

    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_i  = int(probs.argmax())
        print(f"Predicted: {classes[top_i]}  (confidence={probs[top_i]:.3f})")
        for i,c in enumerate(classes):
            print(f"  {c:>5}: {probs[i]:.3f}")

if __name__ == "__main__":
    main()
