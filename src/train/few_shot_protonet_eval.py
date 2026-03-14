"""
MURMUR ProtoNet Evaluation — Two complementary experiments.

Experiment 1: Per-subject personalisation
  For each test subject:
    - Global model predicts on their sample (no personalisation)
    - ProtoNet uses their OWN sample as prototype, predicts same sample
    - Shows per-subject improvement from personalisation

Experiment 2: Cross-subject few-shot generalisation
  For k = 1, 3, 5, 10:
    - For each test subject as query:
        * Sample k DIFFERENT subjects' embeddings as support
        * Build prototype from those k subjects
        * Classify the query subject
    - Repeat N_TRIALS times and average
    - Produces genuine few-shot generalisation curve
"""
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.train.train_ms_tcn_2c import MSTCN, pick_device
from src.train.protonet_cal_2c import embed_batch

# Known global model baseline from full test set eval (murmur-6.ipynb)
GLOBAL_BASELINE_ACC = 0.718
GLOBAL_BASELINE_F1  = 0.716

N_TRIALS = 10  # repeat cross-subject eval N times, average results


def preprocess(path, max_len):
    """Load .npy, apply CMVN, pad/crop. Same as training pipeline."""
    x = np.load(path)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std  = x.std(axis=(1, 2), keepdims=True) + 1e-8
    x    = (x - mean) / std
    T = x.shape[-1]
    if T < max_len:
        pad = np.zeros((3, 64, max_len - T), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=-1)
    else:
        x = x[:, :, :max_len]
    return x


def get_embedding(model, path, max_len, device):
    """Preprocess + embed a single .npy file. Returns (x_np, emb_np)."""
    x_np = preprocess(path, max_len)
    X_t  = torch.from_numpy(x_np).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embed_batch(model, X_t)[0].cpu().numpy()
    return x_np, emb


def cosine_predict(emb, protos, classes):
    """Classify emb against protos dict using cosine similarity."""
    sims = [
        np.dot(emb, protos[c]) /
        (np.linalg.norm(emb) * np.linalg.norm(protos[c]) + 1e-9)
        for c in classes
    ]
    return int(np.argmax(sims))


# ── EXPERIMENT 1: Per-subject personalisation ────────────────────────────────

def experiment1_per_subject(test_df, classes, c2i, model, max_len, device):
    """
    For each test subject, compare global model vs ProtoNet on their sample.
    ProtoNet uses the subject's own embedding as prototype.
    Shows direct per-subject benefit of personalisation.
    """
    subjects = sorted(test_df["subject_id"].unique())
    skipped = 0
    improved = unchanged = degraded = 0

    all_true_g, all_pred_g = [], []
    all_true_p, all_pred_p = [], []

    for sid in subjects:
        sdf = test_df[test_df["subject_id"] == sid]

        class_embs = {}
        class_xnps = {}
        valid = True
        for c in classes:
            rows = sdf[sdf["label"] == c]
            if len(rows) == 0:
                valid = False
                break
            x_np, emb = get_embedding(model, rows.iloc[0]["filepath"], max_len, device)
            class_embs[c] = emb
            class_xnps[c] = x_np

        if not valid:
            skipped += 1
            continue

        protos = {c: class_embs[c] for c in classes}
        sub_g_correct = True
        sub_p_correct = True

        for c in classes:
            yi   = c2i[c]
            x_np = class_xnps[c]
            emb  = class_embs[c]

            # Global model
            X_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_g = int(model(X_t).argmax(1).cpu())

            # ProtoNet
            pred_p = cosine_predict(emb, protos, classes)

            all_true_g.append(yi); all_pred_g.append(pred_g)
            all_true_p.append(yi); all_pred_p.append(pred_p)

            if pred_g != yi: sub_g_correct = False
            if pred_p != yi: sub_p_correct = False

        if sub_p_correct and not sub_g_correct:
            improved += 1
        elif sub_g_correct and not sub_p_correct:
            degraded += 1
        else:
            unchanged += 1

    total = len(subjects) - skipped

    acc_g = accuracy_score(all_true_g, all_pred_g)
    f1_g  = f1_score(all_true_g, all_pred_g, average="macro")
    acc_p = accuracy_score(all_true_p, all_pred_p)
    f1_p  = f1_score(all_true_p, all_pred_p, average="macro")

    print("\n" + "=" * 66)
    print("EXPERIMENT 1: Per-subject personalisation")
    print("=" * 66)
    print(f"Subjects evaluated: {total} | Skipped: {skipped}")
    print(f"\n{'Model':<14} {'Acc':>8} {'Macro F1':>10}")
    print(f"{'-'*34}")
    print(f"{'Global':<14} {acc_g:>8.3f} {f1_g:>10.3f}")
    print(f"{'ProtoNet':<14} {acc_p:>8.3f} {f1_p:>10.3f}")
    print(f"{'ΔF1':<14} {acc_p-acc_g:>+8.3f} {f1_p-f1_g:>+10.3f}")
    print(f"\nPer-subject outcome (out of {total} subjects):")
    print(f"  Improved  (ProtoNet correct, Global wrong): {improved:>4}  ({improved/total*100:.1f}%)")
    print(f"  Unchanged (both correct or both wrong)    : {unchanged:>4}  ({unchanged/total*100:.1f}%)")
    print(f"  Degraded  (Global correct, ProtoNet wrong): {degraded:>4}  ({degraded/total*100:.1f}%)")

    print(f"\nClassification report — Global model:")
    print(classification_report(all_true_g, all_pred_g,
                                target_names=classes, digits=3))
    print(f"Classification report — ProtoNet (1-shot):")
    print(classification_report(all_true_p, all_pred_p,
                                target_names=classes, digits=3))

    return acc_g, f1_g, acc_p, f1_p, improved, unchanged, degraded


# ── EXPERIMENT 2: Cross-subject few-shot generalisation ──────────────────────

def experiment2_cross_subject(test_df, classes, c2i, model, max_len, device,
                               shot_counts=(1, 3, 5, 10), n_trials=N_TRIALS,
                               seed=7):
    """
    For k = 1, 3, 5, 10:
      For each test subject as query:
        - Sample k DIFFERENT subjects' embeddings per class as support
        - Build prototype from those k support subjects
        - Classify the query
      Repeat n_trials times and average F1.
      This tests genuine few-shot generalisation across subjects.
    """
    subjects = sorted(test_df["subject_id"].unique())
    rng = np.random.default_rng(seed)

    # Pre-compute all embeddings once
    print("\nPre-computing embeddings for all test subjects...")
    subject_data = {}
    skipped = 0
    for sid in subjects:
        sdf = test_df[test_df["subject_id"] == sid]
        data = {}
        valid = True
        for c in classes:
            rows = sdf[sdf["label"] == c]
            if len(rows) == 0:
                valid = False
                break
            _, emb = get_embedding(model, rows.iloc[0]["filepath"],
                                   max_len, device)
            data[c] = emb
        if valid:
            subject_data[sid] = data
        else:
            skipped += 1

    valid_subjects = list(subject_data.keys())
    print(f"Valid subjects: {len(valid_subjects)} | Skipped: {skipped}")

    print("\n" + "=" * 66)
    print("EXPERIMENT 2: Cross-subject few-shot generalisation")
    print(f"Global baseline — Acc: {GLOBAL_BASELINE_ACC:.3f} | F1: {GLOBAL_BASELINE_F1:.3f}")
    print("=" * 66)
    print(f"{'Shots':>6} | {'Mean Acc':>9} | {'Mean F1':>8} | "
          f"{'Std F1':>7} | {'ΔF1 vs baseline':>16}")
    print("-" * 56)

    all_results = []
    for k in shot_counts:
        trial_f1s  = []
        trial_accs = []

        for trial in range(n_trials):
            all_true, all_pred = [], []

            for qry_sid in valid_subjects:
                support_pool = [s for s in valid_subjects if s != qry_sid]

                protos = {}
                for c in classes:
                    pool_c   = [s for s in support_pool
                                if c in subject_data[s]]
                    k_actual = min(k, len(pool_c))
                    chosen   = rng.choice(len(pool_c), k_actual,
                                          replace=False)
                    embs     = [subject_data[pool_c[i]][c]
                                for i in chosen]
                    protos[c] = np.mean(embs, axis=0)

                for c in classes:
                    emb  = subject_data[qry_sid][c]
                    pred = cosine_predict(emb, protos, classes)
                    all_true.append(c2i[c])
                    all_pred.append(pred)

            trial_accs.append(accuracy_score(all_true, all_pred))
            trial_f1s.append(f1_score(all_true, all_pred,
                                      average="macro"))

        mean_acc = float(np.mean(trial_accs))
        mean_f1  = float(np.mean(trial_f1s))
        std_f1   = float(np.std(trial_f1s))
        delta    = mean_f1 - GLOBAL_BASELINE_F1

        print(f"{k:>6} | {mean_acc:>9.3f} | {mean_f1:>8.3f} | "
              f"{std_f1:>7.4f} | {delta:>+16.3f}")
        all_results.append((k, mean_acc, mean_f1, std_f1, delta))

    return all_results


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="manifests/split_2c_subjectwise.csv")
    ap.add_argument("--ckpt",      default="models/ms_tcn_colab_1024.pt")
    ap.add_argument("--max_len",   type=int, default=1024)
    ap.add_argument("--n_trials",  type=int, default=N_TRIALS)
    a = ap.parse_args()

    df = pd.read_csv(a.split_csv)
    classes = sorted(df[df.split == "train"]["label"].unique())
    c2i = {c: i for i, c in enumerate(classes)}

    device = pick_device()
    print(f"Device  : {device}")
    print(f"Classes : {classes}")

    model = MSTCN(in_ch=3, n_classes=len(classes)).to(device)
    model.load_state_dict(
        torch.load(a.ckpt, map_location=device), strict=False
    )
    model.eval()

    test_df = df[df.split == "test"].copy()

    experiment1_per_subject(
        test_df, classes, c2i, model, a.max_len, device
    )

    experiment2_cross_subject(
        test_df, classes, c2i, model, a.max_len, device,
        shot_counts=[1, 3, 5, 10], n_trials=a.n_trials
    )


if __name__ == "__main__":
    main()