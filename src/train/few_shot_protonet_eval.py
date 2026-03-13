"""
Few-shot ProtoNet evaluation across multiple shot counts.
Produces the shot-count vs F1 table for the thesis.

Fix: Global model now uses the same CMVN-normalised + padded tensor
as ProtoNet — no raw reload from disk which caused F1=0.333.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from src.train.train_ms_tcn_2c import MSTCN, pick_device
from src.train.protonet_cal_2c import embed_batch


def preprocess(path, max_len):
    """Load .npy, apply CMVN, pad/crop to max_len. Returns numpy array [3,64,max_len]."""
    x = np.load(path)                                   # [3, 64, T]
    mean = x.mean(axis=(1, 2), keepdims=True)
    std  = x.std(axis=(1, 2), keepdims=True) + 1e-8
    x = (x - mean) / std                               # CMVN — same as training
    T = x.shape[-1]
    if T < max_len:
        pad = np.zeros((3, 64, max_len - T), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=-1)
    else:
        x = x[:, :, :max_len]
    return x                                            # [3, 64, max_len]


def run_eval(split_csv, ckpt, shots, max_len, seed=7):
    """
    For each test subject:
      - Preprocess all clips with CMVN + pad/crop (same as training)
      - Use min(shots, available) clips per class as ProtoNet support
      - Remaining clips as queries
      - Classify with both ProtoNet (cosine) and global model (softmax)
      - Both use the IDENTICAL preprocessed tensor — no discrepancy
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
    skipped = 0

    for sid in subjects:
        sdf = test_df[test_df["subject_id"] == sid]

        # ── collect preprocessed arrays + embeddings per class
        class_data = {}   # c -> list of (x_np, embedding_np)
        for c in classes:
            rows = sdf[sdf["label"] == c]
            if len(rows) == 0:
                break
            items = []
            for _, row in rows.iterrows():
                x_np = preprocess(row["filepath"], max_len)          # [3,64,max_len]
                X_t  = torch.from_numpy(x_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = embed_batch(model, X_t)[0].cpu().numpy()   # [D]
                items.append((x_np, emb))
            class_data[c] = items

        if len(class_data) < len(classes):
            skipped += 1
            continue

        # ── build prototypes from support set
        protos = {}
        query_items = {}   # c -> list of (x_np, emb_np)
        for c in classes:
            items = class_data[c]
            k = min(shots, len(items))
            sup_idx = rng.choice(len(items), k, replace=False).tolist()
            qry_idx = [i for i in range(len(items)) if i not in sup_idx]
            if not qry_idx:
                qry_idx = sup_idx[-1:]   # fallback: use last support as query

            protos[c]      = np.mean([items[i][1] for i in sup_idx], axis=0)
            query_items[c] = [items[i] for i in qry_idx]

        # ── classify queries with both models
        for c in classes:
            yi = c2i[c]
            for x_np, emb in query_items[c]:

                # ProtoNet — cosine similarity to prototypes
                sims = [
                    np.dot(emb, protos[cc]) /
                    (np.linalg.norm(emb) * np.linalg.norm(protos[cc]) + 1e-9)
                    for cc in classes
                ]
                all_pred_proto.append(int(np.argmax(sims)))
                all_true_proto.append(yi)

                # Global model — softmax head, SAME preprocessed tensor
                X_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(X_t)
                all_pred_global.append(int(logits.argmax(1).cpu()))
                all_true_global.append(yi)

    print(f"Evaluated: {len(subjects) - skipped} subjects | Skipped: {skipped}")

    acc_g = accuracy_score(all_true_global, all_pred_global)
    f1_g  = f1_score(all_true_global, all_pred_global, average="macro")
    acc_p = accuracy_score(all_true_proto,  all_pred_proto)
    f1_p  = f1_score(all_true_proto,  all_pred_proto,  average="macro")

    return acc_g, f1_g, acc_p, f1_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="manifests/split_2c_subjectwise.csv")
    ap.add_argument("--ckpt",      default="models/ms_tcn_colab_1024.pt")
    ap.add_argument("--max_len",   type=int, default=1024)
    a = ap.parse_args()

    print(f"{'Shots':>6} | {'Global Acc':>10} | {'Global F1':>9} | "
          f"{'Proto Acc':>9} | {'Proto F1':>8} | {'ΔF1':>6}")
    print("-" * 62)

    for k in [1, 3, 5, 10]:
        acc_g, f1_g, acc_p, f1_p = run_eval(
            a.split_csv, a.ckpt, shots=k, max_len=a.max_len
        )
        delta = f1_p - f1_g
        print(f"{k:>6} | {acc_g:>10.3f} | {f1_g:>9.3f} | "
              f"{acc_p:>9.3f} | {f1_p:>8.3f} | {delta:>+6.3f}")


if __name__ == "__main__":
    main()