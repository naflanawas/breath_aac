import argparse, numpy as np, torch, librosa, matplotlib.pyplot as plt
from pathlib import Path
from src.train.train_ms_tcn_2c import MSTCN
import pandas as pd

def mel_delta_stack(y, sr=16000, n_mels=64, n_fft=1024, hop=256, fmin=50, fmax=8000):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                       hop_length=hop, fmin=fmin, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    d1 = librosa.feature.delta(S_db, width=9).astype(np.float32)
    d2 = librosa.feature.delta(S_db, order=2, width=9).astype(np.float32)
    return np.stack([S_db, d1, d2], axis=0)  # [3,64,T]

def fix_len(x, T=256):
    if x.shape[-1] < T:
        pad = np.zeros((x.shape[0], x.shape[1], T - x.shape[-1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=-1)
    return x[:, :, :T]

def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def gradcam_on_wav(wav, ckpt, split_csv, out_png, target_class=None):
    device = pick_device()
    classes = sorted(pd.read_csv(split_csv).query("split=='train'")["label"].unique())

    model = MSTCN(in_ch=3, n_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    feats = {}
    def fwd_hook(module, inp, out):
        feats["act"] = out.detach()   # detach activations
        feats["act"].requires_grad = True
    def bwd_hook(module, grad_in, grad_out):
        feats["grad"] = grad_out[0].detach()

    h1 = model.fuse.register_forward_hook(fwd_hook)
    h2 = model.fuse.register_full_backward_hook(bwd_hook)

    y, sr = librosa.load(wav, sr=16000, mono=True)
    m = np.max(np.abs(y));  y = y/(m+1e-9) if m>0 else y
    X = fix_len(mel_delta_stack(y, sr=sr), 256)
    X_t = torch.from_numpy(X).unsqueeze(0).to(device)  # [1,3,64,256]

    logits = model(X_t)
    probs = torch.softmax(logits, dim=1)[0]
    pred_i = int(probs.argmax())
    cls_i = pred_i if target_class is None else classes.index(target_class)

    model.zero_grad()
    logits[0, cls_i].backward()

    A   = feats["act"][0]              # [C,64,256]
    dA  = feats["grad"][0]             # [C,64,256]
    w   = dA.mean(dim=(1,2))           # [C]
    cam = torch.relu((w[:,None,None] * A).sum(dim=0))  # [64,256]
    cam = cam / (cam.max() + 1e-9)
    cam_np = cam.detach().cpu().numpy()  # <-- detach before numpy

    logmel = X[0]
    sec_per_frame = 256/16000
    extent = [0, logmel.shape[1]*sec_per_frame, 0, logmel.shape[0]]

    plt.figure(figsize=(7,4))
    plt.imshow(logmel, origin="lower", aspect="auto", extent=extent)
    plt.imshow(cam_np, origin="lower", aspect="auto", extent=extent, alpha=0.35)
    plt.title(f"{Path(wav).name}  | pred={classes[pred_i]} ({probs[pred_i]:.2f})")
    plt.xlabel("Time (s)"); plt.ylabel("Mel bins")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    h1.remove(); h2.remove()
    return classes[pred_i], float(probs[pred_i])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", default="models/ms_tcn_2c.pt")
    ap.add_argument("--split_csv", default="manifests/split_2c.csv")
    ap.add_argument("--out", default="viz/gradcam/out.png")
    ap.add_argument("--class", dest="force_class", default=None)
    a = ap.parse_args()
    pred, conf = gradcam_on_wav(a.wav, a.ckpt, a.split_csv, a.out, a.force_class)
    print(f"Saved {a.out}  |  Predicted: {pred} (conf={conf:.3f})")
