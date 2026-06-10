"""
generate_gradcam_viva_v2.py  ―  Smarter-sample Grad-CAM figure for MURMUR viva.

Run from the repo root:
    python generate_gradcam_viva_v2.py

Produces: gradcam_viva_figure_v2.png  (300 dpi, ~14×13 inches)

Key improvement over v1:
  - Correct SHORT  : only considers .npy samples with T ≤ 300 frames (~4.8 s),
                     i.e. genuine compact/single-puff recordings, not long
                     multi-cycle captures.  This gives a focused spectrogram
                     where Grad-CAM saliency is expected in the early-mid frames.
  - Correct LONG   : highest-confidence correctly-classified LONG sample (any T).
                     Multi-cycle recordings are fine here — distributed saliency
                     across the full duration is precisely the expected pattern.
  - Wrong (L → S)  : actual-long sample predicted as short; among candidates
                     the one with the smallest T is preferred so the display
                     resembles an isolated single puff.

All three panels still use the pre-extracted .npy feature files and hook
Grad-CAM into model.fuse (the 1×1 Conv2d that merges the four dilated TCN
branches), identical to v1.
"""

import sys
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train.train_ms_tcn_2c import MSTCN
from src.utils.device import pick_device

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT      = "models/ms_tcn_no_cmvn.pt"
SPLIT_CSV = "manifests/split_2c_subjectwise.csv"
MAX_LEN   = 1024          # frames (matches training)
HOP       = 256           # hop length used in feature extraction
SR        = 16000         # sample rate
OUT_PNG   = "gradcam_viva_figure_v2.png"

# Maximum T for the SHORT panel — enforces a compact, single-puff spectrogram.
SHORT_MAX_T = 300         # ≈ 4.8 s at hop=256, sr=16000


# ── Helpers ───────────────────────────────────────────────────────────────────

def fix_len(x: np.ndarray, T: int = MAX_LEN) -> np.ndarray:
    """Pad with zeros (right) or truncate the time axis to exactly T frames."""
    if x.shape[-1] < T:
        pad = np.zeros((x.shape[0], x.shape[1], T - x.shape[-1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=-1)
    return x[:, :, :T]


def run_gradcam(
    model: torch.nn.Module,
    x_np: np.ndarray,
    target_idx: int,
    device: torch.device,
):
    """
    Compute Grad-CAM at model.fuse for the given sample and target class.

    Hooks into the 1×1 Conv2d fuse layer (output shape B×64×64×T), computes
    gap-weighted activations, applies ReLU, and crops to the natural duration.

    Args:
        x_np:       Natural-length feature array, shape (3, 64, T_nat).
        target_idx: Class index whose logit is used for the backward pass.

    Returns:
        cam_np: Normalised Grad-CAM map, shape (64, T_nat), values in [0, 1].
        probs:  Softmax probabilities from the forward pass, shape (n_classes,).
    """
    T_nat = x_np.shape[-1]
    X_t   = torch.from_numpy(fix_len(x_np)).unsqueeze(0).to(device).float()

    cache = {}

    def _fwd(module, inp, out):
        cache["act"] = out                   # (1, 64, 64, 1024)

    def _bwd(module, grad_in, grad_out):
        cache["grad"] = grad_out[0].detach() # (1, 64, 64, 1024)

    h_fwd = model.fuse.register_forward_hook(_fwd)
    h_bwd = model.fuse.register_full_backward_hook(_bwd)

    model.zero_grad()
    logits = model(X_t)                             # forward pass (eval mode)
    probs  = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    logits[0, target_idx].backward()               # backward for Grad-CAM

    h_fwd.remove()
    h_bwd.remove()

    A   = cache["act"][0]                           # (64, 64, 1024)
    dA  = cache["grad"][0]                          # (64, 64, 1024)
    w   = dA.mean(dim=(1, 2))                       # GAP → (64,)
    cam = torch.relu((w[:, None, None] * A).sum(dim=0))  # (64, 1024)
    cam = cam / (cam.max() + 1e-9)
    cam_np = cam.detach().cpu().numpy()[:, :T_nat]  # crop to natural length

    return cam_np, probs


def saliency_label(cam_np: np.ndarray) -> str:
    """
    Describe where the saliency mass is concentrated, in plain English.
    Uses the temporal centre-of-mass and spread of the Grad-CAM energy.
    """
    t_profile = cam_np.mean(axis=0)        # (T_nat,)
    T     = len(t_profile)
    ts    = np.arange(T, dtype=float) / T  # normalised [0, 1]
    total = t_profile.sum() + 1e-9
    com   = float(np.dot(ts, t_profile) / total)
    var   = float(np.dot((ts - com) ** 2, t_profile) / total)
    spread = np.sqrt(var)

    if spread > 0.20:
        return "Saliency: distributed across full duration"
    if com < 0.35:
        return "Saliency: concentrated in early frames"
    if com > 0.65:
        return "Saliency: concentrated in later frames"
    return "Saliency: concentrated in mid-breath"


# ── Sample selection ───────────────────────────────────────────────────────────

def find_samples(model, test_df: pd.DataFrame, classes: list, device):
    """
    Iterate over the test set and return three representative samples:

      1. Highest-confidence correctly-predicted SHORT puff with T ≤ SHORT_MAX_T
         (genuine compact recording, not a long multi-cycle capture).
      2. Highest-confidence correctly-predicted LONG puff (any T; the full-
         duration distributed saliency is the expected and illustrative result).
      3. First (smallest-T) long puff misclassified as short — ideally an
         isolated single puff that looks similar to a short.

    Each returned value is a tuple:
        (filepath, x_np, true_label, pred_label, conf, T_nat)
    """
    best_short = (None, -1.0)       # (entry, conf) for compact correct short
    best_long  = (None, -1.0)       # (entry, conf) for correct long
    wrong_candidates = []           # all long→short misclassifications

    print("\nScanning test set for representative samples...")
    for _, row in test_df.iterrows():
        fp    = row["filepath"]
        label = row["label"]
        x_np  = np.load(fp)                            # (3, 64, T_nat)
        T_nat = x_np.shape[-1]

        X_t = torch.from_numpy(fix_len(x_np)).unsqueeze(0).to(device).float()
        with torch.no_grad():
            probs = torch.softmax(model(X_t), dim=1)[0].cpu().numpy()

        pred_cl = classes[int(probs.argmax())]
        conf    = float(probs.max())
        entry   = (fp, x_np, label, pred_cl, conf, T_nat)

        # Correct SHORT: enforce compact T to avoid multi-cycle recordings
        if label == "short" and pred_cl == "short" and T_nat <= SHORT_MAX_T:
            if conf > best_short[1]:
                best_short = (entry, conf)

        # Correct LONG: any T, highest confidence
        if label == "long" and pred_cl == "long" and conf > best_long[1]:
            best_long = (entry, conf)

        # Wrong (long → short): collect all, then pick smallest T
        if label == "long" and pred_cl == "short":
            wrong_candidates.append(entry)

    assert best_short[0]       is not None, (
        f"No correctly-predicted short puff with T≤{SHORT_MAX_T} found in test set"
    )
    assert best_long[0]        is not None, "No correct long puff found in test set"
    assert len(wrong_candidates) > 0,       "No long→short misclassification found"

    # For the wrong panel prefer the smallest T (most isolated-looking puff)
    wrong_long_as_short = min(wrong_candidates, key=lambda e: e[5])

    cs = best_short[0]
    cl = best_long[0]
    cw = wrong_long_as_short
    print(f"  ✓ Correct SHORT : conf={cs[4]:.3f}  T={cs[5]}  {cs[0]}")
    print(f"  ✓ Correct LONG  : conf={cl[4]:.3f}  T={cl[5]}  {cl[0]}")
    print(f"  ✗ Wrong (L→S)   : conf={cw[4]:.3f}  T={cw[5]}  {cw[0]}")
    return cs, cl, cw


# ── Figure builder ─────────────────────────────────────────────────────────────

def build_figure(samples, model, classes, device) -> plt.Figure:
    """
    Build the 3-row × 2-column Grad-CAM figure.

    Layout uses a 6-row GridSpec: even rows are thin title rows, odd rows
    contain the (log-Mel | Grad-CAM) panel pairs.
    """
    c2i = {c: i for i, c in enumerate(classes)}

    cases = [
        (samples[0], "Correct — Short Puff",                          c2i["short"]),
        (samples[1], "Correct — Long Puff",                           c2i["long"]),
        (samples[2], "Wrong Prediction — Actual Long, Predicted Short", c2i["short"]),
    ]

    N   = len(cases)
    fig = plt.figure(figsize=(14, 13), facecolor="white")

    gs = gridspec.GridSpec(
        2 * N, 2,
        figure=fig,
        height_ratios=[0.06, 1] * N,
        hspace=0.10,
        wspace=0.06,
        top=0.94, bottom=0.05,
        left=0.06, right=0.96,
    )

    for i, ((fp, x_np, true_cl, pred_cl, conf, T_nat),
            row_title, target_idx) in enumerate(cases):

        cam_np, probs = run_gradcam(model, x_np, target_idx, device)

        logmel  = x_np[0, :, :T_nat]       # (64, T_nat)  —  channel 0 = log-Mel
        dur_sec = T_nat * HOP / SR
        extent  = [0, dur_sec, 0, 64]
        ann     = saliency_label(cam_np)

        # ── Title strip (spans both columns) ─────────────────────────────────
        ax_t = fig.add_subplot(gs[2 * i, :])
        ax_t.set_axis_off()
        conf_label = (f"   |   pred = {pred_cl},  conf = {conf:.0%}"
                      f",  T = {T_nat} frames  ({dur_sec:.1f} s)")
        ax_t.text(
            0.5, 0.45,
            row_title + conf_label,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            transform=ax_t.transAxes,
        )

        # ── Left panel: plain log-Mel spectrogram ────────────────────────────
        ax_mel = fig.add_subplot(gs[2 * i + 1, 0])
        ax_mel.imshow(
            logmel, origin="lower", aspect="auto",
            extent=extent, cmap="magma", interpolation="nearest",
        )
        ax_mel.set_xlabel("Time (s)", fontsize=9)
        ax_mel.set_ylabel("Mel bins",  fontsize=9)
        ax_mel.tick_params(labelsize=8)
        ax_mel.set_title("Log-Mel spectrogram", fontsize=9, pad=3)

        # ── Right panel: log-Mel + Grad-CAM overlay ──────────────────────────
        ax_cam = fig.add_subplot(gs[2 * i + 1, 1])
        ax_cam.imshow(
            logmel, origin="lower", aspect="auto",
            extent=extent, cmap="magma", interpolation="nearest",
        )
        hm = ax_cam.imshow(
            cam_np, origin="lower", aspect="auto",
            extent=extent, cmap="inferno", alpha=0.55,
            vmin=0, vmax=1, interpolation="bilinear",
        )
        ax_cam.set_xlabel("Time (s)", fontsize=9)
        ax_cam.tick_params(labelsize=8, labelleft=False)
        ax_cam.set_title("Grad-CAM overlay  (fuse layer, 1×1 conv)", fontsize=9, pad=3)

        # Saliency annotation box (bottom-right of Grad-CAM panel)
        ax_cam.text(
            0.97, 0.04, ann,
            transform=ax_cam.transAxes,
            ha="right", va="bottom", fontsize=8,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="#000000", alpha=0.60),
        )

        # Colourbar
        cbar = fig.colorbar(hm, ax=ax_cam, fraction=0.046, pad=0.04, aspect=20)
        cbar.set_label("Saliency", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Figure-level title
    fig.text(
        0.5, 0.975,
        "Grad-CAM Visualisations — MURMUR MS-TCN  "
        "(hook: model.fuse, no CMVN, test set)",
        ha="center", va="top", fontsize=13, fontweight="bold",
    )

    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    device = pick_device()
    print(f"Device  : {device}")

    df      = pd.read_csv(SPLIT_CSV)
    classes = sorted(df[df.split == "train"]["label"].unique())
    print(f"Classes : {classes}")          # ['long', 'short']

    # Load model (eval mode; gradients remain enabled for Grad-CAM backward)
    model = MSTCN(in_ch=3, n_classes=len(classes)).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded  : {CKPT}")

    test_df = df[df.split == "test"].copy().reset_index(drop=True)
    print(f"Test samples: {len(test_df)}")

    correct_short, correct_long, wrong = find_samples(
        model, test_df, classes, device
    )

    print("\nGenerating Grad-CAM figure...")
    fig = build_figure([correct_short, correct_long, wrong], model, classes, device)

    Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {OUT_PNG}  (300 dpi)")


if __name__ == "__main__":
    main()
