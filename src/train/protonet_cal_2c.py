import argparse, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.train.train_ms_tcn_2c import MSTCN, MelClipSet
from src.utils.device import pick_device

def embed_batch(model, X):
    """Extract intermediate embeddings from MSTCN (before the classifier head).
 
    Runs the stem → branches → fuse → pool pipeline and returns the flattened
    feature vector.  Used by ProtoNet for prototype building and similarity scoring.
 
    Args:
        model: MSTCN instance.
        X: Input tensor of shape (B, in_ch, n_mels, T).
 
    Returns:
        Tensor of shape (B, D) where D is the pool output dimension.
    """
    h = model.stem(X)
    feats = [b(h) for b in model.branches]
    h = torch.cat(feats, dim=1)
    h = model.fuse(h)
    h = model.pool(h)
    h = h.view(h.size(0), -1)
    return h            

def few_shot_eval(split_csv, ckpt, shots=5, max_len=256, seed=7):
    """
    True few-shot personalisation eval.
    For each TEST subject:
      - use min(shots, available) clips per class as support
      - remaining clips as queries
    Compares ProtoNet vs global model on same queries.
    """
    df = pd.read_csv(split_csv)
    classes = sorted(df[df.split == "train"]["label"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    device = pick_device()

    model = MSTCN(in_ch=3, n_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()

    test_df = df[df.split == "test"].copy()
    subjects = sorted(test_df["subject_id"].unique())

    rng = np.random.default_rng(seed)

    all_true_global, all_pred_global = [], []
    all_true_proto,  all_pred_proto  = [], []

    def load_embed(path):
        """Load a .npy feature file, apply CMVN + padding, and return its embedding."""
        x = np.load(path)                          # [3, 64, T]
        mean = x.mean(axis=(1,2), keepdims=True)
        std  = x.std(axis=(1,2), keepdims=True) + 1e-8
        x = (x - mean) / std
        T = x.shape[-1]
        if T < max_len:
            pad = np.zeros((3, 64, max_len - T), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=-1)
        else:
            x = x[:, :, :max_len]
        X = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = embed_batch(model, X)            # [1, D]
        return emb[0].cpu().numpy()

    skipped = 0
    for sid in subjects:
        sdf = test_df[test_df["subject_id"] == sid]

        # collect embeddings per class
        class_embs = {}
        for c in classes:
            rows = sdf[sdf["label"] == c]
            if len(rows) == 0:
                break
            embs = [load_embed(r["filepath"]) for _, r in rows.iterrows()]
            class_embs[c] = embs

        if len(class_embs) < len(classes):
            skipped += 1
            continue

        # build prototype from first 'shots' embeddings
        protos = {}
        queries = {}
        for c in classes:
            embs = class_embs[c]
            k = min(shots, len(embs))
            support_idx = rng.choice(len(embs), k, replace=False)
            query_idx   = [i for i in range(len(embs)) if i not in support_idx]
            protos[c]   = np.mean([embs[i] for i in support_idx], axis=0)
            queries[c]  = [embs[i] for i in query_idx] if query_idx else [embs[support_idx[-1]]]

        # classify queries
        for c in classes:
            yi = c2i[c]
            for e in queries[c]:
                # ProtoNet (cosine)
                sims = [np.dot(e, protos[cc]) /
                        (np.linalg.norm(e) * np.linalg.norm(protos[cc]) + 1e-9)
                        for cc in classes]
                all_pred_proto.append(int(np.argmax(sims)))
                all_true_proto.append(yi)

                # Global model (softmax head)
                x_t = torch.from_numpy(
                    np.load(sdf[sdf["label"] == c].iloc[0]["filepath"])
                ).unsqueeze(0).to(device)
                # pad/crop
                if x_t.shape[-1] < max_len:
                    pad = torch.zeros(1, 3, 64, max_len - x_t.shape[-1])
                    x_t = torch.cat([x_t, pad], dim=-1)
                else:
                    x_t = x_t[:, :, :, :max_len]
                with torch.no_grad():
                    logits = model(x_t)
                all_pred_global.append(int(logits.argmax(1).cpu()))
                all_true_global.append(yi)

    print(f"Evaluated: {len(subjects)-skipped} subjects | Skipped: {skipped}")

    acc_g = accuracy_score(all_true_global, all_pred_global)
    f1_g  = f1_score(all_true_global, all_pred_global, average="macro")
    acc_p = accuracy_score(all_true_proto, all_pred_proto)
    f1_p  = f1_score(all_true_proto,  all_pred_proto,  average="macro")

    return acc_g, f1_g, acc_p, f1_p
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="manifests/split_2c.csv")
    ap.add_argument("--ckpt", default="models/ms_tcn_2c.pt")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=256)
    a = ap.parse_args()
    acc_g, f1_g, acc_p, f1_p = few_shot_eval(a.split_csv, a.ckpt, shots=a.shots, max_len=a.max_len)
    print(f"Global   acc={acc_g:.3f}  f1={f1_g:.3f}")
    print(f"ProtoNet {a.shots}-shot  acc={acc_p:.3f}  f1={f1_p:.3f}")
