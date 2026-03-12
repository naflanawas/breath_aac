import argparse, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.train.train_ms_tcn_2c import MSTCN, MelClipSet

def embed_batch(model, X):
    h = model.stem(X)
    feats = [b(h) for b in model.branches]
    h = torch.cat(feats, dim=1)
    h = model.fuse(h)
    h = model.pool(h)
    h = h.view(h.size(0), -1)
    return h               # [B, D]

def few_shot_eval(split_csv, ckpt, shots=5, max_len=256, seed=7):
    df = pd.read_csv(split_csv)
    classes = sorted(df[df.split=="train"]["label"].unique())
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # pick 'shots' samples per class from TRAIN as support; evaluate on VAL as queries
    tr = df[df.split=="train"].copy()
    va = df[df.split=="val"].copy()

    support_rows=[]
    for c in classes:
        pool = tr[tr.label==c]
        take = min(shots, len(pool))
        support_rows.append(pool.sample(take, random_state=seed))
    support = pd.concat(support_rows)

    sup_set = MelClipSet(split_csv, "train", max_len=max_len, classes=classes)
    qry_set = MelClipSet(split_csv, "test",   max_len=max_len, classes=classes)

    sup_idx = sup_set.df.index[sup_set.df.filepath.isin(support.filepath)]
    sup_loader = DataLoader(torch.utils.data.Subset(sup_set, sup_idx), batch_size=32, shuffle=False)
    qry_loader = DataLoader(qry_set, batch_size=32, shuffle=False)

    model = MSTCN(in_ch=3, n_classes=len(classes)).to(device)
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # build class prototypes (mean embedding per class)
    protos = {}
    with torch.no_grad():
        for X,y in sup_loader:
            X=X.to(device); y=y.to(device)
            E = embed_batch(model, X)  # [B, D]
            for e, yi in zip(E, y):
                c = classes[int(yi)]
                protos.setdefault(c, []).append(e.cpu().numpy())
    for c in protos:
        protos[c] = np.mean(np.stack(protos[c], axis=0), axis=0)

    # classify VAL by nearest prototype (cosine similarity)
    ys=[]; ps=[]
    with torch.no_grad():
        for X,y in qry_loader:
            X=X.to(device); y=y.to(device)
            E = embed_batch(model, X).cpu().numpy()
            for e, yi in zip(E, y.cpu().numpy()):
                sims=[np.dot(e,protos[c])/(np.linalg.norm(e)*np.linalg.norm(protos[c])+1e-9) for c in classes]
                ps.append(int(np.argmax(sims))); ys.append(int(yi))

    acc = accuracy_score(ys, ps)
    f1  = f1_score(ys, ps, average="macro")
    return acc, f1

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="manifests/split_2c.csv")
    ap.add_argument("--ckpt", default="models/ms_tcn_2c.pt")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=256)
    a = ap.parse_args()
    acc, f1 = few_shot_eval(a.split_csv, a.ckpt, shots=a.shots, max_len=a.max_len)
    print(f"ProtoNet {a.shots}-shot  VAL acc={acc:.3f}  f1={f1:.3f}")
