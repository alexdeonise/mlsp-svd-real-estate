import argparse
import csv
import os

import numpy as np

from svd_common import compute_svd, format_svd_info


ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--A",
        required=True
    )

    ap.add_argument(
        "--out",
        required=True
    )

    ap.add_argument(
        "--solver",
        choices=["custom", "numpy"],
        default="numpy"
    )

    ap.add_argument(
        "--rtol",
        type=float,
        default=None
    )

    ap.add_argument(
        "--atol",
        type=float,
        default=None
    )

    a = ap.parse_args()

    A = np.load(a.A)

    U, s, Vt, info = compute_svd(
        A,
        solver=a.solver,
        rtol=a.rtol,
        atol=a.atol
    )

    print(format_svd_info(info))

    numerical_rank = info["rank"]

    energy = s ** 2

    info_kept = np.cumsum(energy) / np.sum(energy)

    remaining = np.clip(
        1.0 - info_kept,
        0.0,
        1.0
    )

    rel_err = np.sqrt(remaining)

    k90 = int(
        np.searchsorted(
            info_kept,
            0.90
        ) + 1
    )

    k95 = int(
        np.searchsorted(
            info_kept,
            0.95
        ) + 1
    )

    os.makedirs(
        os.path.dirname(a.out),
        exist_ok=True
    )

    with open(
        a.out,
        "w",
        newline=""
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "k",
            "info_kept",
            "rel_reconstruction_error"
        ])

        for k in range(
            1,
            len(s) + 1
        ):
            w.writerow([
                k,
                f"{info_kept[k - 1]:.6f}",
                f"{rel_err[k - 1]:.6f}"
            ])

    print("shape:", A.shape)
    print("numerical rank:", numerical_rank)
    print("k pentru 90%:", k90)
    print("k pentru 95%:", k95)
    print("scris:", a.out)


if __name__ == "__main__":
    main()