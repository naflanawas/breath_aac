import argparse, pathlib, pandas as pd, numpy as np, librosa, soundfile as sf
from scipy.signal import butter, lfilter

SHORT_MS = (400, 1000)     # 0.4–1.0 s
LONG_MS  = (1800, 3000)    # 1.8–3.0 s

def bp(y, sr, lo=200, hi=4000, order=4):
    b,a = butter(order, [lo/(sr/2), hi/(sr/2)], btype='band'); return lfilter(b,a,y)

def bursts_from_rms(y, sr, hop=256, win=512, k=2.5, min_len_ms=250, merge_gap_ms=200):
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop).squeeze()
    med = np.median(rms); mad = np.median(np.abs(rms-med)) + 1e-8
    thr = med + k*mad
    act = rms > thr
    idx = np.where(np.diff(np.concatenate([[0], act.view(np.int8), [0]])) != 0)[0]
    spans = [(idx[i], idx[i+1]) for i in range(0,len(idx),2)]
    hop_ms = 1000*hop/sr
    merged=[]
    for s,e in spans:
        if merged and (s*hop_ms-merged[-1][1]*hop_ms) < merge_gap_ms:
            merged[-1]=(merged[-1][0], e)
        else:
            merged.append((s,e))
    return [(int(s*hop_ms), int(e*hop_ms)) for s,e in merged if (e-s)*hop_ms >= min_len_ms]

def label_from_length(ms):
    if SHORT_MS[0] <= ms <= SHORT_MS[1]: return "short"
    if LONG_MS[0]  <= ms <= LONG_MS[1]:  return "long"
    return None

def crop_center_ms(y, sr, start_ms, end_ms, target_ms):
    dur = end_ms - start_ms
    if dur <= target_ms:
        ssm = int(start_ms*sr/1000); eem = int(end_ms*sr/1000)
        return y[ssm:eem]
    mid = (start_ms + end_ms) // 2
    half = target_ms // 2
    s = max(0, int((mid - half)*sr/1000)); e = min(len(y), int((mid + half)*sr/1000))
    return y[s:e]

def deterministic_slice(y, sr, want="short"):
    target_ms = 800 if want=="short" else 2500
    Tms = int(len(y)*1000/sr)
    if Tms <= target_ms+200:
        return y
    start_ms = max(0, (Tms - target_ms)//2)
    end_ms   = start_ms + target_ms
    return crop_center_ms(y, sr, start_ms, end_ms, target_ms)

def run(in_root, out_root, manifest_csv, sr=16000):
    in_root, out_root = pathlib.Path(in_root), pathlib.Path(out_root)
    rows=[]
    for p in in_root.rglob("*.wav"):
        y,_ = librosa.load(p.as_posix(), sr=sr, mono=True)
        y = bp(y, sr)
        bursts = bursts_from_rms(y, sr)
        label=None; clip=None
        if bursts:
            s,e = max(bursts, key=lambda t: t[1]-t[0])
            dur = e - s
            label = label_from_length(dur)
            if label:
                tgt = 800 if label=="short" else 2500
                clip = crop_center_ms(y, sr, s, e, tgt)
        if label is None:
            name = p.name.lower()
            if "deep" in name:   label, clip = "long",  deterministic_slice(y, sr, "long")
            elif "shallow" in name: label, clip = "short", deterministic_slice(y, sr, "short")
            else:
                continue
        outp = out_root/label/f"{p.stem}.wav"
        outp.parent.mkdir(parents=True, exist_ok=True)
        sf.write(outp.as_posix(), clip, sr)
        rows.append([str(outp), label, p.name, p.parent.name])
    pd.DataFrame(rows, columns=["filepath","label","source_file","source_dir"]).to_csv(manifest_csv, index=False)
    print(f"Wrote {manifest_csv} with {len(rows)} items.")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root",  required=True)      # e.g. data_std/custom
    ap.add_argument("--out_root", required=True)      # e.g. clips_2c
    ap.add_argument("--manifest", default="manifests/segments_2c.csv")
    ap.add_argument("--sr", type=int, default=16000)
    a = ap.parse_args()
    run(a.in_root, a.out_root, a.manifest, sr=a.sr)
