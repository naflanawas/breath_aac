import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── EDIT THIS PATH ──────────────────────────────────────────────
VERIFICATION_DIR = os.path.expanduser("~/Documents/augmentation_verification")
OUT_DIR          = os.path.expanduser("~/Documents/augmentation_verification")
# ────────────────────────────────────────────────────────────────

VARIANTS = [
    "1_original",
    "2_louder",
    "3_quieter",
    "4_timeshift",
    "5_noise",
]

VARIANT_LABELS = {
    "1_original":  "1 Original",
    "2_louder":    "2 Louder",
    "3_quieter":   "3 Quieter",
    "4_timeshift": "4 Time-shift",
    "5_noise":     "5 Noise",
}

GESTURES = {
    "short": "Short Puff",
    "long":  "Long Puff",
}

def make_diagram(subject_folder_name, gesture_key, gesture_label, out_dir):
    subject_dir = os.path.join(VERIFICATION_DIR, subject_folder_name)
    
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'Augmentation Verification — {gesture_label} — {subject_folder_name}\n'
        f'All 5 variants from same subject (confirmed test set, never used in training)',
        fontsize=12, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.3)

    for col, variant_key in enumerate(VARIANTS):
        # File name is simple, just short_1_original.wav
        filename = f"{gesture_key}_{variant_key}.wav"
        path = os.path.join(subject_dir, filename)

        if not os.path.exists(path):
            print(f"  WARNING: File not found: {path}")
            continue

        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y[:, 0]
        y = y.astype(np.float32)
        times = np.linspace(0, len(y) / sr, len(y))

        # --- Waveform ---
        ax_wave = fig.add_subplot(gs[0, col])
        ax_wave.plot(times, y, color='#1f77b4', linewidth=0.6, alpha=0.85)
        ax_wave.set_title(VARIANT_LABELS[variant_key], fontsize=10,
                          fontweight='bold', pad=4)
        ax_wave.set_ylim(-1.05, 1.05)
        ax_wave.set_xlabel('Time (s)', fontsize=8)
        if col == 0:
            ax_wave.set_ylabel('Amplitude', fontsize=8)
        ax_wave.tick_params(labelsize=7)
        ax_wave.axhline(0, color='gray', linewidth=0.4, linestyle='--')
        rms = np.sqrt(np.mean(y ** 2))
        ax_wave.text(0.97, 0.93, f'RMS={rms:.4f}',
                     transform=ax_wave.transAxes, fontsize=7,
                     ha='right', va='top', color='#d62728',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor='#d62728', alpha=0.8))

        # --- Mel Spectrogram ---
        ax_mel = fig.add_subplot(gs[1, col])
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, n_fft=1024,
            hop_length=256, fmin=50, fmax=8000
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(
            S_db, sr=sr, hop_length=256,
            x_axis='time', y_axis='mel',
            fmin=50, fmax=8000,
            ax=ax_mel, cmap='magma'
        )
        ax_mel.tick_params(labelsize=7)
        ax_mel.set_xlabel('Time (s)', fontsize=8)
        if col == 0:
            ax_mel.set_ylabel('Frequency (Hz)', fontsize=8)
        else:
            ax_mel.set_ylabel('')

    fig.text(0.01, 0.73, 'Waveform', va='center', rotation='vertical',
             fontsize=10, color='#333333', fontweight='bold')
    fig.text(0.01, 0.28, 'Mel\nSpectrogram', va='center', rotation='vertical',
             fontsize=10, color='#333333', fontweight='bold')

    out_name = f"diagram_{subject_folder_name}_{gesture_key}.png"
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_name}")

os.makedirs(OUT_DIR, exist_ok=True)

# Dynamically find all subject folders
subject_folders = [d for d in os.listdir(VERIFICATION_DIR) if os.path.isdir(os.path.join(VERIFICATION_DIR, d)) and d.startswith("subject")]
subject_folders.sort()

if not subject_folders:
    print(f"No subject folders found in {VERIFICATION_DIR}")
    exit(1)

count = 0
for sf_name in subject_folders:
    for gesture_key, gesture_label in GESTURES.items():
        print(f"Generating diagram for: {sf_name} — {gesture_label}...")
        make_diagram(sf_name, gesture_key, gesture_label, OUT_DIR)
        count += 1

print(f"\nDone. {count} diagrams saved to: {OUT_DIR}")