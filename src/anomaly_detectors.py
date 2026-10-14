"""Trei scoruri de anomalie pe A; anomalie = scor peste percentila --pct.

    python src/anomaly_detectors.py --methods kmeans iforest --pct 95
    python src/anomaly_detectors.py --methods svd kmeans iforest --pct 95 --k 5
    python src/anomaly_detectors.py --methods svd --solver numpy --pct 95
Scrie results/anomalies_<metoda>.csv: row, score, flagged
"""
import argparse, os, csv, numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from svd_common import compute_svd, format_svd_info

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def score_svd(A, k, solver="custom"):
    U, s, Vt, info = compute_svd(A, solver=solver)          # format comun: s vector, Vt transpus
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    return np.sum((A - Ak)**2, axis=1), info

def score_kmeans(A, k=4, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(A)
    return np.linalg.norm(A - km.cluster_centers_[km.labels_], axis=1)

def score_iforest(A, seed=42):
    iso = IsolationForest(contamination=0.05, random_state=seed).fit(A)
    return -iso.score_samples(A)                            # mai mare = mai anormal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["kmeans","iforest"], choices=["svd","kmeans","iforest"])
    ap.add_argument("--pct", type=float, default=95); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--solver", choices=["custom","numpy"], default="custom")   # custom = SVD-ul Mariei, ca in baseline
    ap.add_argument("--A", default=os.path.join(ROOT, "data", "A.npy")); ap.add_argument("--tag", default="")
    a = ap.parse_args(); A = np.load(a.A)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    for m in a.methods:
        if m == "svd":
            sc, info = score_svd(A, a.k, solver=a.solver)
            print(format_svd_info(info))
        else:
            sc = {"kmeans": lambda: score_kmeans(A), "iforest": lambda: score_iforest(A)}[m]()
        thr = np.percentile(sc, a.pct); flag = sc > thr
        out = os.path.join(ROOT, "results", f"anomalies_{m}{a.tag}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["row","score","flagged"])
            for i, (s_, fl) in enumerate(zip(sc, flag)): w.writerow([i, f"{s_:.6f}", int(fl)])
        print(f"{m:<8} prag p{a.pct:g}={thr:.4f}  marcate={int(flag.sum())}/{len(A)}  -> {os.path.relpath(out, ROOT)}")

if __name__ == "__main__":
    main()
