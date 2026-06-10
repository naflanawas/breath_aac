#!/usr/bin/env python3
"""
gradcam_from_wav.py  ―  Batch Grad-CAM comparison figure from three WAV files.

Run from the repo root:
    python gradcam_from_wav.py <wav1> <wav2> <wav3>

Produces: gradcam_wav_comparison.png  (150 dpi)

Three rows, one per WAV, each with the same three panels as live_gradcam_demo.py:
    Left   — log-Mel spectrogram (librosa.display.specshow)
    Middle — same spectrogram + Grad-CAM overlay + saliency annotation
    Right  — result card  (predicted class, confidence, duration, tips)
"""

import sys
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train.train_ms_tcn_2c import MSTCN
from src.utils.device import pick_device

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT    = "models/ms_tcn_no_cmvn.pt"
SR      = 16_000
N_MELS  = 64
N_FFT   = 1024
HOP     = 256
FMIN    = 50
FMAX    = 8000
MAX_LEN = 1024
OUT_PNG = "gradcam_wav_comparison.png"

# VAD parameters — identical to live_gradcam_demo.py / inference BreathDetector
VAD_FRAME  = 512
VAD_HOP    = 256
VAD_THRESH = 0.02
VAD_MIN_MS = 100
VAD_MAX_MS = 3000

# Classes sorted alphabetically (matches training label sort)
CLASSES = ["long", "short"]   # index 0 = long, index 1 = short

# Per-row context labels supplied by the user
ROW_LABELS = [
    "Long Breath  (mobile: 80% correct)",
    "Short Breath  (mobile: 100% correct)",
    "Ambiguous — Short+Short pattern  (mobile: I NEED WATER  73%)",
]


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> torch.nn.Module:
    model = MSTCN(in_ch=3, n_classes=len(CLASSES)).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── VAD — identical to live_gradcam_demo.py ───────────────────────────────────

def vad_best_segment(audio: np.ndarray) -> np.ndarray | None:
    """Peak-normalise and apply STE-based VAD with 15-frame closing buffer.
    Returns the longest valid segment waveform, or None."""
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / (peak + 1e-9)

    energy = np.array([
        float(np.sum(audio[i : i + VAD_FRAME] ** 2))
        for i in range(0, len(audio) - VAD_FRAME, VAD_HOP)
    ], dtype=np.float32)

    if energy.max() < 0.005:
        return None
    energy /= energy.max()

    active = energy > VAD_THRESH

    smoothed = active.copy()
    below = 0
    for i in range(len(active)):
        if active[i]:
            below = 0
        else:
            below += 1
            if below <= 15:
                smoothed[i] = True

    min_s = int(VAD_MIN_MS * SR / 1000)
    max_s = int(VAD_MAX_MS * SR / 1000)

    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, a in enumerate(smoothed):
        if a and not in_seg:
            seg_start = i * VAD_HOP
            in_seg = True
        elif not a and in_seg:
            seg_end = i * VAD_HOP
            in_seg = False
            if min_s <= (seg_end - seg_start) <= max_s:
                segments.append((seg_start, seg_end))

    if in_seg:
        seg_end = len(audio)
        if min_s <= (seg_end - seg_start) <= max_s:
            segments.append((seg_start, seg_end))

    if not segments:
        return None

    best = max(segments, key=lambda se: se[1] - se[0])
    return audio[best[0] : best[1]]


# ── Features — identical to live_gradcam_demo.py ──────────────────────────────

def extract_features(y: np.ndarray) -> np.ndarray | None:
    """Log-Mel + Δ + ΔΔ at natural length. Returns (3, 64, T_nat) or None."""
    if len(y) < 9 * HOP:
        return None
    mel    = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    logmel  = librosa.power_to_db(mel, ref=np.max)
    delta   = librosa.feature.delta(logmel, width=9)
    delta2  = librosa.feature.delta(logmel, order=2, width=9)
    return np.stack([logmel, delta, delta2], axis=0).astype(np.float32)


def fix_len(x: np.ndarray, T: int = MAX_LEN) -> np.ndarray:
    """Pad with zeros (right) or truncate time axis to exactly T frames."""
    if x.shape[-1] < T:
        pad = np.zeros((x.shape[0], x.shape[1], T - x.shape[-1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=-1)
    return x[:, :, :T]


# ── Grad-CAM — identical to live_gradcam_demo.py ──────────────────────────────

def run_gradcam(
    model: torch.nn.Module,
    x_np: np.ndarray,
    target_idx: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM at model.fuse. Returns (cam_np [0,1], probs)."""
    T_nat = x_np.shape[-1]
    X_t   = torch.from_numpy(fix_len(x_np)).unsqueeze(0).to(device).float()

    cache: dict = {}

    def _fwd(module, inp, out):
        cache["act"] = out

    def _bwd(module, grad_in, grad_out):
        cache["grad"] = grad_out[0].detach()

    h_fwd = model.fuse.register_forward_hook(_fwd)
    h_bwd = model.fuse.register_full_backward_hook(_bwd)

    model.zero_grad()
    logits = model(X_t)
    probs  = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    logits[0, target_idx].backward()

    h_fwd.remove()
    h_bwd.remove()

    A   = cache["act"][0]
    dA  = cache["grad"][0]
    w   = dA.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * A).sum(dim=0))
    cam = cam / (cam.max() + 1e-9)
    cam_np = cam.detach().cpu().numpy()[:, :T_nat]

    return cam_np, probs


def saliency_text(pred_class: str) -> str:
    if pred_class == "short":
        return "Saliency: early onset region"
    return "Saliency: sustained across duration"


# ── Per-row pipeline ──────────────────────────────────────────────────────────

def process_wav(wav_path: str, model: torch.nn.Module, device: torch.device) -> dict:
    """Load WAV → VAD → features → Grad-CAM. Returns result dict."""
    print(f"\n  Processing: {Path(wav_path).name}")

    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    print(f"    Loaded: {len(y)/SR:.2f}s  peak={np.abs(y).max():.3f}")

    y_seg = vad_best_segment(y)
    if y_seg is None:
        raise RuntimeError(f"No valid VAD segment found in {wav_path}")

    dur_ms = int(len(y_seg) / SR * 1000)
    print(f"    VAD segment: {dur_ms} ms")

    x_np = extract_features(y_seg)
    if x_np is None:
        raise RuntimeError(f"Segment too short for feature extraction in {wav_path}")

    # Inference
    X_t = torch.from_numpy(fix_len(x_np)).unsqueeze(0).to(device).float()
    with torch.no_grad():
        probs_det = torch.softmax(model(X_t), dim=1)[0].cpu().numpy()

    pred_idx   = int(probs_det.argmax())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs_det[pred_idx])
    print(f"    Prediction: {pred_class.upper()}  ({confidence*100:.1f}%)")

    # Grad-CAM
    cam_np, _ = run_gradcam(model, x_np, pred_idx, device)
    print(f"    CAM shape: {cam_np.shape}  range=[{cam_np.min():.3f}, {cam_np.max():.3f}]")

    return dict(
        x_np=x_np, cam_np=cam_np,
        pred_class=pred_class, confidence=confidence, dur_ms=dur_ms,
    )


# ── Figure builder ────────────────────────────────────────────────────────────

def draw_row(
    fig: plt.Figure,
    axes,                       # (ax_mel, ax_cam, ax_txt)
    row_label: str,
    x_np: np.ndarray,
    cam_np: np.ndarray,
    pred_class: str,
    confidence: float,
    dur_ms: int,
) -> None:
    """Fill one row (3 axes) with mel spec / Grad-CAM overlay / result card."""
    logmel = x_np[0]            # (64, T_nat)

    ax_mel, ax_cam, ax_txt = axes

    # ── Left: plain log-Mel spectrogram ──────────────────────────────────────
    librosa.display.specshow(
        logmel, sr=SR, hop_length=HOP,
        x_axis="time", y_axis="mel",
        fmin=FMIN, fmax=FMAX,
        cmap="magma", ax=ax_mel,
    )
    # Row label as the spectrogram panel title (bold, slightly larger)
    ax_mel.set_title(row_label, fontsize=11, fontweight="bold", pad=5, loc="left")
    ax_mel.set_xlabel("Time (s)", fontsize=9)
    ax_mel.set_ylabel("Frequency (Hz)", fontsize=9)
    ax_mel.tick_params(labelsize=8)

    # ── Middle: log-Mel + Grad-CAM overlay ───────────────────────────────────
    librosa.display.specshow(
        logmel, sr=SR, hop_length=HOP,
        x_axis="time", y_axis="mel",
        fmin=FMIN, fmax=FMAX,
        cmap="magma", ax=ax_cam,
    )
    x0, x1 = ax_cam.get_xlim()
    y0, y1 = ax_cam.get_ylim()

    hm = ax_cam.imshow(
        cam_np,
        origin="lower", aspect="auto",
        extent=[x0, x1, y0, y1],
        cmap="inferno", alpha=0.5,
        vmin=0, vmax=1,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(hm, ax=ax_cam, fraction=0.046, pad=0.04, aspect=18)
    cbar.set_label("Saliency", fontsize=8)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7)

    ax_cam.set_title("Grad-CAM  (hook: model.fuse)", fontsize=10, pad=5)
    ax_cam.set_xlabel("Time (s)", fontsize=9)
    ax_cam.set_ylabel("")
    ax_cam.tick_params(labelleft=False, labelsize=8)

    ax_cam.text(
        0.97, 0.04,
        saliency_text(pred_class),
        transform=ax_cam.transAxes,
        ha="right", va="bottom", fontsize=8,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc="black", alpha=0.65),
    )

    # ── Right: result card ────────────────────────────────────────────────────
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.set_axis_off()

    label_str = "SHORT PUFF" if pred_class == "short" else "LONG PUFF"
    label_col = "#c0510a"    if pred_class == "short" else "#0055aa"
    conf_str  = f"Confidence: {confidence * 100:.1f}%"
    expl_str  = ("Brief early energy burst detected"
                 if pred_class == "short"
                 else "Sustained breath energy\nacross duration detected")

    ax_txt.text(0.5, 0.90, label_str,
                transform=ax_txt.transAxes, ha="center", va="top",
                fontsize=22, fontweight="bold", color=label_col)

    ax_txt.text(0.5, 0.72, conf_str,
                transform=ax_txt.transAxes, ha="center", va="top",
                fontsize=13, color="#222222")

    ax_txt.text(0.5, 0.60, expl_str,
                transform=ax_txt.transAxes, ha="center", va="top",
                fontsize=10, color="#555555", style="italic")

    sep = Line2D([0.05, 0.95], [0.49, 0.49],
                 transform=ax_txt.transAxes,
                 color="#cccccc", linewidth=1.2)
    ax_txt.add_artist(sep)

    ax_txt.text(0.5, 0.44, f"Segment: {dur_ms} ms",
                transform=ax_txt.transAxes, ha="center", va="top",
                fontsize=10, color="#333333")

    ax_txt.text(0.5, 0.34,
                "Check if saliency is on the\nbreath region, not silence.",
                transform=ax_txt.transAxes, ha="center", va="top",
                fontsize=10, color="#333333")

    ax_txt.text(0.5, 0.06,
                "Saliency on noise → prediction\nmay be unreliable.",
                transform=ax_txt.transAxes, ha="center", va="bottom",
                fontsize=8, style="italic", color="#888888")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python gradcam_from_wav.py <wav1> <wav2> <wav3>")
        sys.exit(1)

    wav_paths = sys.argv[1:4]

    # Verify files exist
    for p in wav_paths:
        if not Path(p).exists():
            print(f"Error: file not found — {p}")
            sys.exit(1)

    device = pick_device()
    print("─" * 62)
    print("Loading MURMUR MS-TCN model…")
    model = load_model(device)
    print(f"  Model loaded  device={device}")
    print("─" * 62)

    # ── Run pipeline for each WAV ─────────────────────────────────────────────
    results = []
    for wav in wav_paths:
        results.append(process_wav(wav, model, device))

    # ── Build combined 3-row figure ───────────────────────────────────────────
    print("\nBuilding figure…")

    N_ROWS = 3
    fig = plt.figure(figsize=(18, 17), facecolor="white")
    fig.suptitle(
        "MURMUR — Grad-CAM Analysis  (hook: model.fuse, no CMVN, VAD-isolated segments)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    # GridSpec: 3 rows, 3 columns, with a thin horizontal gap between rows
    gs = gridspec.GridSpec(
        N_ROWS, 3,
        figure=fig,
        hspace=0.42,      # vertical gap between rows
        wspace=0.08,      # horizontal gap between columns
        top=0.96, bottom=0.04,
        left=0.05, right=0.97,
        width_ratios=[2, 2, 1],   # spec | cam | card
    )

    for row_idx, (res, label) in enumerate(zip(results, ROW_LABELS)):
        ax_mel = fig.add_subplot(gs[row_idx, 0])
        ax_cam = fig.add_subplot(gs[row_idx, 1])
        ax_txt = fig.add_subplot(gs[row_idx, 2])

        draw_row(
            fig,
            (ax_mel, ax_cam, ax_txt),
            row_label=f"Row {row_idx+1}:  {label}",
            x_np=res["x_np"],
            cam_np=res["cam_np"],
            pred_class=res["pred_class"],
            confidence=res["confidence"],
            dur_ms=res["dur_ms"],
        )

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓  Saved → {OUT_PNG}  (150 dpi)")


if __name__ == "__main__":
    main()
