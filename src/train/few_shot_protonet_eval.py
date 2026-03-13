"""
Few-shot ProtoNet evaluation across multiple shot counts.
Produces the shot-count vs F1 table for the thesis.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.train.protonet_cal_2c import few_shot_eval

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
        acc_g, f1_g, acc_p, f1_p = few_shot_eval(
            a.split_csv, a.ckpt, shots=k, max_len=a.max_len
        )
        delta = f1_p - f1_g
        print(f"{k:>6} | {acc_g:>10.3f} | {f1_g:>9.3f} | "
              f"{acc_p:>9.3f} | {f1_p:>8.3f} | {delta:>+6.3f}")

if __name__ == "__main__":
    main()