#To visualize the exported mel images
# View Mel / Delta / Delta-Delta saved as [3, 64, T] .npy
import argparse, glob, os, numpy as np, matplotlib.pyplot as plt

def show_one(npy_path, sr=16000, hop=256, out_dir="viz"):
    x = np.load(npy_path)  # shape: (3, 64, T)
    os.makedirs(out_dir, exist_ok=True)
    sec_per_frame = hop / sr
    T = x.shape[-1]
    extent = [0, T*sec_per_frame, 0, x.shape[1]]  # x in seconds, y in mel bins

    titles = ["Log-Mel", "Delta (Δ)", "Delta-Delta (ΔΔ)"]
    outs   = ["mel_log.png", "mel_delta.png", "mel_deltadelta.png"]

    for i in range(3):
        plt.figure()
        im = plt.imshow(x[i], aspect="auto", origin="lower", extent=extent, interpolation="nearest")
        plt.title(f"{titles[i]} — {os.path.basename(npy_path)}")
        plt.xlabel("Time (s)")
        plt.ylabel("Mel bins")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        outp = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(npy_path))[0]}_{outs[i]}")
        plt.savefig(outp, dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved:", outp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", help="Path to a single .npy")
    ap.add_argument("--glob", help="Glob to search (e.g., 'features/mel_dd/*/*.npy')")
    args = ap.parse_args()

    if args.npy:
        show_one(args.npy)
    else:
        paths = sorted(glob.glob(args.glob or "features/mel_dd/*/*.npy"))
        if not paths:
            print("No .npy found. Try: --npy <file> or --glob 'features/mel_dd/*/*.npy'")
        else:
            show_one(paths[0])
