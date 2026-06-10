#!/usr/bin/env python3
"""
live_gradcam_demo.py  ―  Real-time Grad-CAM demo for MURMUR viva presentation.

Run from the repo root:
    python live_gradcam_demo.py

What it does
  1. Loads the MS-TCN model once; keeps it in memory for every recording.
  2. Records from the default microphone at 16 kHz until ENTER is pressed.
  3. Applies STE-based VAD (identical to inference BreathDetector) to isolate
     the best breath segment from the recording.
  4. Extracts log-Mel + Δ + ΔΔ features and runs forward/backward pass.
  5. Computes Grad-CAM at model.fuse (the 1×1 Conv2d that merges the four
     dilated TCN branches).
  6. Saves a 3-panel figure (mel spectrogram | Grad-CAM overlay | result card)
     as live_demo_result.png and opens it automatically.
  7. Prints a plain-English error-analysis summary to the terminal.
  8. Loops until the user types Q.
"""

# ── Auto-install sounddevice if missing ───────────────────────────────────────
import subprocess
import sys

def _ensure_pkg(pkg: str) -> None:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[setup] {pkg} not found — installing via pip...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[setup] {pkg} installed.")

_ensure_pkg("sounddevice")

# ── Standard imports ──────────────────────────────────────────────────────────
import threading
import platform
import numpy as np
import torch
import librosa
import librosa.display
import sounddevice as sd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Allow `from src...` imports when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train.train_ms_tcn_2c import MSTCN
from src.utils.device import pick_device

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT     = "models/ms_tcn_no_cmvn.pt"
SR       = 16_000        # sample rate
N_MELS   = 64
N_FFT    = 1024
HOP      = 256
FMIN     = 50
FMAX     = 8000
MAX_LEN  = 1024          # model input length in frames (pad/crop)
REC_CHUNK = 1024         # sounddevice read chunk in samples
OUT_PNG  = "live_demo_result.png"

# VAD parameters — identical to inference BreathDetector
VAD_FRAME  = 512
VAD_HOP    = 256
VAD_THRESH = 0.02
VAD_MIN_MS = 100
VAD_MAX_MS = 3000

# Classes sorted alphabetically (matches training label sort)
CLASSES = ["long", "short"]   # index 0 = long, index 1 = short


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> torch.nn.Module:
    model = MSTCN(in_ch=3, n_classes=len(CLASSES)).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── Recording ─────────────────────────────────────────────────────────────────

def record_until_enter() -> np.ndarray:
    """
    Stream from the default microphone at SR Hz (mono, float32) until the user
    presses ENTER.  A daemon thread waits on input() while sounddevice collects
    audio chunks; once ENTER is detected the stream closes cleanly.

    Returns: 1-D float32 numpy array of the full recording.
    """
    print("\nRecording... blow into the mic. Press ENTER to stop.")

    stop_event = threading.Event()
    chunks: list[np.ndarray] = []

    def _wait_for_enter() -> None:
        input()                 # blocks until ENTER
        stop_event.set()

    listener = threading.Thread(target=_wait_for_enter, daemon=True)
    listener.start()

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=REC_CHUNK) as stream:
        while not stop_event.is_set():
            data, _ = stream.read(REC_CHUNK)
            chunks.append(data.flatten())

    audio = np.concatenate(chunks, dtype=np.float32)
    print(f"  ↳ Captured {len(audio) / SR:.2f} s  ({len(audio):,} samples)")
    return audio


# ── VAD ───────────────────────────────────────────────────────────────────────

def vad_best_segment(audio: np.ndarray) -> np.ndarray | None:
    """
    Peak-normalise and apply STE-based VAD.  Uses a 15-frame closing buffer
    (≈150 ms) to handle gradual breath fade-outs.

    Returns the *longest* valid segment waveform, or None if none found.
    """
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / (peak + 1e-9)

    # Short-time energy
    energy = np.array([
        float(np.sum(audio[i : i + VAD_FRAME] ** 2))
        for i in range(0, len(audio) - VAD_FRAME, VAD_HOP)
    ], dtype=np.float32)

    if energy.max() < 0.005:          # pure silence
        return None
    energy /= energy.max()

    active = energy > VAD_THRESH

    # Closing buffer: keep a segment open for 15 consecutive below-threshold
    # frames before closing it (~150 ms at hop=256, sr=16000).
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

    # Return the longest segment found
    best = max(segments, key=lambda se: se[1] - se[0])
    return audio[best[0] : best[1]]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(y: np.ndarray) -> np.ndarray | None:
    """
    Log-Mel + Δ + ΔΔ features, identical to the training pipeline.
    No CMVN.  Returns (3, 64, T_nat) float32 array, or None if too short.
    """
    if len(y) < 9 * HOP:    # librosa.feature.delta requires width=9 frames
        return None

    mel  = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    logmel  = librosa.power_to_db(mel, ref=np.max)   # (64, T_nat)
    delta   = librosa.feature.delta(logmel, width=9)
    delta2  = librosa.feature.delta(logmel, order=2, width=9)

    return np.stack([logmel, delta, delta2], axis=0).astype(np.float32)


def fix_len(x: np.ndarray, T: int = MAX_LEN) -> np.ndarray:
    """Pad with zeros (right) or truncate time axis to exactly T frames."""
    if x.shape[-1] < T:
        pad = np.zeros((x.shape[0], x.shape[1], T - x.shape[-1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=-1)
    return x[:, :, :T]


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def run_gradcam(
    model: torch.nn.Module,
    x_np: np.ndarray,
    target_idx: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Grad-CAM at model.fuse (Conv2d(256, 64, 1)).

    The fuse layer output has shape (B, 64, 64, 1024).  Channel weights are
    obtained by global average pooling the gradients over the spatial (mel-bin)
    and temporal axes:  w_c = (1/HW) Σ_{i,t} dScore/dA_{c,i,t}.

    Returns:
        cam_np : normalised saliency map, shape (64, T_nat), values in [0, 1].
        probs  : softmax class probabilities, shape (n_classes,).
    """
    T_nat = x_np.shape[-1]
    X_t   = torch.from_numpy(fix_len(x_np)).unsqueeze(0).to(device).float()

    cache: dict = {}

    def _fwd(module, inp, out):
        cache["act"] = out                        # (1, 64, 64, 1024)

    def _bwd(module, grad_in, grad_out):
        cache["grad"] = grad_out[0].detach()      # (1, 64, 64, 1024)

    h_fwd = model.fuse.register_forward_hook(_fwd)
    h_bwd = model.fuse.register_full_backward_hook(_bwd)

    model.zero_grad()
    logits = model(X_t)
    probs  = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    logits[0, target_idx].backward()

    h_fwd.remove()
    h_bwd.remove()

    A   = cache["act"][0]                         # (64, 64, 1024)
    dA  = cache["grad"][0]                        # (64, 64, 1024)
    w   = dA.mean(dim=(1, 2))                     # GAP over (mel, time) → (64,)
    cam = torch.relu((w[:, None, None] * A).sum(dim=0))  # (64, 1024)
    cam = cam / (cam.max() + 1e-9)
    cam_np = cam.detach().cpu().numpy()[:, :T_nat]

    return cam_np, probs


def saliency_text(pred_class: str) -> str:
    if pred_class == "short":
        return "Saliency: early onset region"
    return "Saliency: sustained across duration"


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(
    x_np: np.ndarray,
    cam_np: np.ndarray,
    pred_class: str,
    confidence: float,
    dur_ms: int,
) -> plt.Figure:
    """
    3-panel figure:
      Left   — plain log-Mel spectrogram (librosa.display.specshow)
      Middle — same spectrogram + Grad-CAM overlay + saliency annotation
      Right  — text result card (label, confidence, explanation, tips)
    """
    T_nat   = x_np.shape[-1]
    logmel  = x_np[0]               # (64, T_nat) — channel 0 = log-Mel

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    fig.suptitle(
        "MURMUR — Live Grad-CAM Error Analysis Demo",
        fontsize=16, fontweight="bold", y=1.02,
    )

    # ── Left panel: plain log-Mel ─────────────────────────────────────────────
    ax_mel = axes[0]
    librosa.display.specshow(
        logmel, sr=SR, hop_length=HOP,
        x_axis="time", y_axis="mel",
        fmin=FMIN, fmax=FMAX,
        cmap="magma", ax=ax_mel,
    )
    ax_mel.set_title("Your Breath — Mel Spectrogram",
                     fontsize=12, fontweight="bold", pad=6)
    ax_mel.set_xlabel("Time (s)", fontsize=10)
    ax_mel.set_ylabel("Frequency (Hz)", fontsize=10)

    # ── Middle panel: log-Mel + Grad-CAM overlay ──────────────────────────────
    ax_cam = axes[1]
    librosa.display.specshow(
        logmel, sr=SR, hop_length=HOP,
        x_axis="time", y_axis="mel",
        fmin=FMIN, fmax=FMAX,
        cmap="magma", ax=ax_cam,
    )
    # Read the exact axis limits that specshow set so the overlay aligns.
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
    cbar = fig.colorbar(hm, ax=ax_cam, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label("Saliency", fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=8)

    ax_cam.set_title("Grad-CAM — What the Model Focused On",
                     fontsize=12, fontweight="bold", pad=6)
    ax_cam.set_xlabel("Time (s)", fontsize=10)
    ax_cam.set_ylabel("")
    ax_cam.tick_params(labelleft=False)

    # Saliency annotation box — bottom right corner of the Grad-CAM panel
    ax_cam.text(
        0.97, 0.04,
        saliency_text(pred_class),
        transform=ax_cam.transAxes,
        ha="right", va="bottom", fontsize=9,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.65),
    )

    # ── Right panel: result card (text only, no axes) ─────────────────────────
    ax_txt = axes[2]
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.set_axis_off()

    label_str  = "SHORT PUFF" if pred_class == "short" else "LONG PUFF"
    label_col  = "#c0510a" if pred_class == "short" else "#0055aa"
    conf_str   = f"Confidence: {confidence * 100:.1f}%"

    if pred_class == "short":
        expl_str = "Brief early energy burst detected"
    else:
        expl_str = "Sustained breath energy\nacross duration detected"

    # Large bold predicted class label
    ax_txt.text(
        0.5, 0.90, label_str,
        transform=ax_txt.transAxes,
        ha="center", va="top",
        fontsize=26, fontweight="bold", color=label_col,
    )
    # Confidence
    ax_txt.text(
        0.5, 0.72, conf_str,
        transform=ax_txt.transAxes,
        ha="center", va="top",
        fontsize=15, color="#222222",
    )
    # Plain-language explanation
    ax_txt.text(
        0.5, 0.60, expl_str,
        transform=ax_txt.transAxes,
        ha="center", va="top",
        fontsize=12, color="#555555", style="italic",
    )
    # Separator line (using axes-coordinate Line2D)
    sep = Line2D(
        [0.05, 0.95], [0.49, 0.49],
        transform=ax_txt.transAxes,
        color="#cccccc", linewidth=1.5,
    )
    ax_txt.add_artist(sep)

    # Segment duration
    ax_txt.text(
        0.5, 0.44, f"Segment duration: {dur_ms} ms",
        transform=ax_txt.transAxes,
        ha="center", va="top",
        fontsize=11, color="#333333",
    )
    # Analysis tip
    ax_txt.text(
        0.5, 0.34,
        "Use this to check if the model\nfocused on the right region.",
        transform=ax_txt.transAxes,
        ha="center", va="top",
        fontsize=11, color="#333333",
    )
    # Small italic warning at the bottom
    ax_txt.text(
        0.5, 0.06,
        "If saliency is on silence or noise\n— prediction may be unreliable.",
        transform=ax_txt.transAxes,
        ha="center", va="bottom",
        fontsize=9, style="italic", color="#888888",
    )

    fig.tight_layout(pad=2.0)
    return fig


# ── Image opener ──────────────────────────────────────────────────────────────

def open_image(path: str) -> None:
    """Open an image file with the default viewer (Mac: open, Linux: xdg-open)."""
    if platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    device = pick_device()
    print("─" * 62)
    print("Loading MURMUR MS-TCN model...")
    model = load_model(device)
    print("MURMUR Live Demo — model loaded")
    print(f"  Device  : {device}")
    print(f"  Classes : {CLASSES}  (index 0 = long, index 1 = short)")
    print(f"  Output  : {OUT_PNG}")
    print("─" * 62)

    while True:
        # ── Record ────────────────────────────────────────────────────────────
        audio = record_until_enter()

        if len(audio) < SR * 0.1:
            print("Recording too short (< 100 ms) — please try again.")
            continue

        # ── VAD ───────────────────────────────────────────────────────────────
        print("Running VAD to isolate breath segment...")
        y_seg = vad_best_segment(audio)

        if y_seg is None:
            print("No breath detected — no segment passed VAD filter.")
            print("Try speaking or breathing louder, or move closer to the mic.")
            continue

        dur_ms = int(len(y_seg) / SR * 1000)
        print(f"  ↳ Breath segment: {dur_ms} ms  ({len(y_seg):,} samples)")

        # ── Feature extraction ────────────────────────────────────────────────
        print("Extracting log-Mel + Δ + ΔΔ features...")
        x_np = extract_features(y_seg)

        if x_np is None:
            print("Segment too short for delta computation — please try again.")
            continue

        T_nat = x_np.shape[-1]
        print(f"  ↳ Feature shape: {x_np.shape}  (natural T = {T_nat} frames)")

        # ── Inference ─────────────────────────────────────────────────────────
        print("Running inference...")
        X_t = torch.from_numpy(fix_len(x_np)).unsqueeze(0).to(device).float()
        with torch.no_grad():
            probs_det = torch.softmax(model(X_t), dim=1)[0].cpu().numpy()

        pred_idx   = int(probs_det.argmax())
        pred_class = CLASSES[pred_idx]
        confidence = float(probs_det[pred_idx])
        print(f"  ↳ Prediction: {pred_class.upper()}  ({confidence * 100:.1f}%)")

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        print("Computing Grad-CAM (hook: model.fuse)...")
        cam_np, _ = run_gradcam(model, x_np, pred_idx, device)
        print(f"  ↳ CAM shape: {cam_np.shape}  (range [{cam_np.min():.3f}, {cam_np.max():.3f}])")

        # ── Figure ────────────────────────────────────────────────────────────
        print("Generating figure...")
        fig = build_figure(x_np, cam_np, pred_class, confidence, dur_ms)
        fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ↳ Figure saved → {OUT_PNG}")

        open_image(OUT_PNG)

        # ── Terminal summary ──────────────────────────────────────────────────
        print()
        print("═" * 62)
        print(f"  Predicted class   : {pred_class.upper()}")
        print(f"  Confidence        : {confidence * 100:.1f}%")
        print(f"  Segment duration  : {dur_ms} ms")
        print("  Correct? If wrong — check which region Grad-CAM highlighted.")
        print("  That tells you why the model failed.")
        print("═" * 62)

        # ── Loop control ──────────────────────────────────────────────────────
        print()
        print("Press ENTER to record again or type Q then ENTER to quit.")
        user_in = input().strip().upper()
        if user_in == "Q":
            print("Exiting MURMUR Live Demo. Goodbye!")
            break


if __name__ == "__main__":
    main()
