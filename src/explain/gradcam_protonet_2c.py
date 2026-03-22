import argparse, numpy as np, torch, librosa, matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from src.train.train_ms_tcn_2c import MSTCN
from src.train.protonet_cal_2c import embed_batch
from src.utils.device import pick_device
from src.features.mel_delta import mel_delta_features as mel_delta_stack

def fix_len(x, T=256):
    """Pad (zeros, right) or truncate the time axis of a feature array to exactly T frames."""
    if x.shape[-1] < T:
        pad = np.zeros((x.shape[0], x.shape[1], T-x.shape[-1]), dtype=x.dtype)
        return np.concatenate([x,pad], axis=-1)
    return x[:,:,:T]

def build_prototypes(model, split_csv, shots=5, max_len=256):
    """Build per-class prototype embeddings from training-set support examples.
 
    Args:
        model: Loaded MSTCN instance (used to extract embeddings).
        split_csv: Path to the split CSV; only ``split=='train'`` rows are used.
        shots: Maximum number of support examples per class.
        max_len: Temporal length used for padding/cropping.
 
    Returns:
        Tuple (classes, protos) where ``classes`` is a sorted list of class
        names and ``protos`` is a dict mapping each class name to a mean
        embedding vector (numpy array).
    """
    df = pd.read_csv(split_csv)
    classes = sorted(df[df.split=="train"]["label"].unique())

    protos = {c:[] for c in classes}

    for c in classes:

        subset = df[(df.split=="train") & (df.label==c)]
        subset = subset.sample(min(shots,len(subset)))

        for _,row in subset.iterrows():

            x = np.load(row["filepath"])
            x = fix_len(x,max_len)

            X = torch.from_numpy(x).unsqueeze(0).to(next(model.parameters()).device)

            with torch.no_grad():
                e = embed_batch(model,X)

            protos[c].append(e.cpu().numpy()[0])

    for c in protos:
        protos[c] = np.mean(protos[c],axis=0)

    return classes, protos

def gradcam_protonet(wav, ckpt, split_csv, out_png):
    """Run ProtoNet-guided Grad-CAM on a WAV file.
 
    Computes class-activation maps using the cosine similarity between the
    query embedding and the nearest prototype as the backward signal.
 
    Args:
        wav: Path to input WAV file.
        ckpt: Path to the trained MS-TCN checkpoint (.pt).
        split_csv: Split CSV used during training (for prototype construction).
        out_png: Output path for the Grad-CAM visualisation.
 
    Returns:
        Predicted class name.
    """
    device = pick_device()

    model = MSTCN(in_ch=3, n_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device))
    model.eval()

    classes, protos = build_prototypes(model,split_csv)

    feats = {}

    def fwd_hook(module,inp,out):
        """Cache forward activations from the fuse layer."""
        feats["act"] = out

    def bwd_hook(module,grad_in,grad_out):
        """Cache gradients flowing back through the fuse layer."""
        feats["grad"] = grad_out[0]

    h1 = model.fuse.register_forward_hook(fwd_hook)
    h2 = model.fuse.register_full_backward_hook(bwd_hook)

    y,sr = librosa.load(wav,sr=16000)

    m = np.max(np.abs(y))
    y = y/(m+1e-9) if m>0 else y

    X = fix_len(mel_delta_stack(y,sr),256)

    X_t = torch.from_numpy(X).unsqueeze(0).to(device)

    emb = embed_batch(model,X_t)

    emb_np = emb.detach().cpu().numpy()[0]

    sims=[]

    for c in classes:

        proto = protos[c]

        s = np.dot(emb_np,proto)/(np.linalg.norm(emb_np)*np.linalg.norm(proto)+1e-9)

        sims.append(s)

    pred_i = int(np.argmax(sims))

    proto = torch.tensor(protos[classes[pred_i]],device=device)

    emb = embed_batch(model,X_t)

    score = torch.nn.functional.cosine_similarity(emb,proto.unsqueeze(0))

    model.zero_grad()

    score.backward()

    A = feats["act"][0]
    dA = feats["grad"][0]

    w = dA.mean(dim=(1,2))

    cam = torch.relu((w[:,None,None]*A).sum(dim=0))

    cam = cam/(cam.max()+1e-9)

    cam_np = cam.detach().cpu().numpy()

    logmel = X[0]

    sec_per_frame = 256/16000

    extent = [0,logmel.shape[1]*sec_per_frame,0,logmel.shape[0]]

    plt.figure(figsize=(7,4))

    plt.imshow(logmel,origin="lower",aspect="auto",extent=extent)

    plt.imshow(cam_np,origin="lower",aspect="auto",extent=extent,alpha=0.35)

    plt.title(f"{Path(wav).name} | ProtoNet pred={classes[pred_i]}")

    plt.xlabel("Time (s)")
    plt.ylabel("Mel bins")

    Path(out_png).parent.mkdir(parents=True,exist_ok=True)

    plt.tight_layout()

    plt.savefig(out_png,dpi=160)

    plt.close()

    h1.remove()
    h2.remove()

    return classes[pred_i]

if __name__=="__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--wav",required=True)
    ap.add_argument("--ckpt",default="models/ms_tcn_2c.pt")
    ap.add_argument("--split_csv",default="manifests/split_2c.csv")
    ap.add_argument("--out",default="viz/gradcam/protonet.png")

    a = ap.parse_args()

    pred = gradcam_protonet(a.wav,a.ckpt,a.split_csv,a.out)

    print("ProtoNet prediction:",pred)