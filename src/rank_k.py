"""Energia retinuta de descompunerea SVD trunchiata la rangul k (M3).

Pentru fiecare matrice (completa si numerica) si fiecare solver (custom —
implementarea principala, si numpy — linia de baza) calculeaza:
  - energia cumulata retinuta la k = 5, 10, 20, 50;
  - rangurile k90 / k95 care pastreaza 90% / 95% din energie;
  - rangul numeric la pragul comun de toleranta.

Iesiri:
  results/rank_k_full.csv       curba completa (custom), matricea completa
  results/rank_k_numeric.csv    curba completa (custom), matricea numerica
  results/rank_k_summary.csv    tabel de comparatie custom vs numpy

Rapoarte la stdout energia la fiecare k de interes. k=5 ramane baseline-ul
original (David il foloseste la detectorul SVD), dar aici il evaluam alaturi
de ranguri care retin cu adevarat semnalul.
"""

import argparse
import csv
import os

import numpy as np

from svd_common import compute_svd, format_svd_info

ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
)

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

PROBE_KS = (5, 10, 20, 50)


def energy_curve(s):
    """Energia cumulata si eroarea relativa de reconstructie pentru fiecare k."""
    energy = s ** 2

    total = np.sum(energy)

    kept = np.cumsum(energy) / total

    remaining = np.clip(
        1.0 - kept,
        0.0,
        1.0,
    )

    rel_err = np.sqrt(remaining)

    return kept, rel_err


def first_k_at(kept, threshold):
    return int(
        np.searchsorted(kept, threshold) + 1
    )


def write_curve(path, kept, rel_err):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow([
            "k",
            "info_kept",
            "rel_reconstruction_error",
        ])

        for k in range(1, len(kept) + 1):
            w.writerow([
                k,
                f"{kept[k - 1]:.6f}",
                f"{rel_err[k - 1]:.6f}",
            ])


def analyze_matrix(name, A, rtol, atol, solvers):
    per_solver = {}

    for solver in solvers:
        U, s, Vt, info = compute_svd(
            A,
            solver=solver,
            rtol=rtol,
            atol=atol,
        )

        print(format_svd_info(info))

        kept, rel_err = energy_curve(s)

        per_solver[solver] = {
            "s": s,
            "kept": kept,
            "rel_err": rel_err,
            "info": info,
        }

    return per_solver


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--full",
        default=os.path.join(DATA, "A.npy"),
    )

    ap.add_argument(
        "--numeric",
        default=os.path.join(DATA, "A_numeric.npy"),
    )

    ap.add_argument(
        "--solver",
        choices=["custom", "numpy", "both"],
        default="both",
    )

    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help=(
            "Prag comun de rang, deasupra etajului de zgomot al metodei "
            "Gram (A^T A); fara el custom raporteaza 393 vs 387 pentru "
            "numpy la convenția eps*max(m,n). Energia cumulativa NU "
            "depinde de acest prag."
        ),
    )

    ap.add_argument(
        "--atol",
        type=float,
        default=None,
    )

    args = ap.parse_args()

    if args.solver == "both":
        solvers = ["custom", "numpy"]
    else:
        solvers = [args.solver]

    matrices = [
        ("full", np.load(args.full)),
        ("numeric", np.load(args.numeric)),
    ]

    os.makedirs(RESULTS, exist_ok=True)

    summary_rows = []

    for name, A in matrices:
        print(f"=== matricea {name}: {A.shape[0]}x{A.shape[1]} ===")

        per_solver = analyze_matrix(
            name,
            A,
            args.rtol,
            args.atol,
            solvers,
        )

        kept_custom = per_solver["custom"]["kept"]
        rel_custom = per_solver["custom"]["rel_err"]

        out_curve = os.path.join(
            RESULTS,
            f"rank_k_{name}.csv",
        )

        write_curve(
            out_curve,
            kept_custom,
            rel_custom,
        )

        print("scris:", out_curve)

        s_custom = per_solver["custom"]["s"]
        s_numpy = per_solver["numpy"]["s"]
        kmin = min(len(s_custom), len(s_numpy))

        row = {
            "matrix": name,
            "m": A.shape[0],
            "n": A.shape[1],
            "rtol": args.rtol,
            "atol": args.atol,
        }

        for solver in solvers:
            data = per_solver[solver]

            kept = data["kept"]
            info = data["info"]

            row[f"rank_{solver}"] = info["rank"]

            row[f"cutoff_{solver}"] = (
                f"{info['cutoff']:.3e}"
            )

            row[f"k90_{solver}"] = first_k_at(
                kept,
                0.90,
            )
            row[f"k95_{solver}"] = first_k_at(
                kept,
                0.95,
            )

            for k in PROBE_KS:
                if k <= len(kept):
                    row[
                        f"energy_k{k}_{solver}"
                    ] = f"{kept[k - 1]:.6f}"
                    row[
                        f"relerr_k{k}_{solver}"
                    ] = f"{data['rel_err'][k - 1]:.6f}"

            print(
                f"[{solver}] rank={info['rank']}",
                f"k90={row[f'k90_{solver}']}",
                f"k95={row[f'k95_{solver}']}",
                " ".join(
                    f"E(k={k})={kept[k - 1]:.4f}"
                    for k in PROBE_KS
                    if k <= len(kept)
                ),
            )

        for k in PROBE_KS:
            if k <= kmin:
                row[f"energy_k{k}_abs_diff"] = (
                    f"{abs(kept_custom[k - 1] - per_solver['numpy']['kept'][k - 1]):.3e}"
                )

        summary_rows.append(row)

    summary_path = os.path.join(
        RESULTS,
        "rank_k_summary.csv",
    )

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )

        w.writeheader()
        w.writerows(summary_rows)

    print("scris:", summary_path)


if __name__ == "__main__":
    main()
