"""Compara solverul custom de SVD cu numpy.linalg.svd pe matricile reale.

Pentru fiecare matrice (completa si numerica) ruleaza ambele solver-e prin
`svd_common.compute_svd` cu ACELEASI tolerante (rtol/atol identice), apoi
compara: valorile singulare, rangul numeric, eroarea relativa de
reconstructie, ortogonalitatea factorilor si predictiile de regresie
(least squares si ridge alpha=1000, pe pret brut si pe log(1+pret)).

Aceasta e validarea ceruta in M1: daca cele doua implementari produc
rezultate echivalente atunci cand impart acelasi prag de rang si aceleasi
tolerante numerice. Scrie results/solver_comparison.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    os.path.join(
        ROOT,
        "src"
    )
)

from svd_common import compute_svd  # noqa: E402

DATA = os.path.join(
    ROOT,
    "data"
)

RESULTS = os.path.join(
    ROOT,
    "results"
)

RIDGE_ALPHA = 1000.0


def reconstruction_error(
    A,
    U,
    s,
    Vt,
):
    return float(
        np.linalg.norm(
            A - (U * s) @ Vt
        )
        / np.linalg.norm(A)
    )


def fit_predict(
    A,
    U,
    s,
    Vt,
    y_fit,
    alpha=None,
):
    y_mean = np.mean(y_fit)
    y_centered = y_fit - y_mean

    projected = U.T @ y_centered

    if alpha is None:
        coef_space = projected / s
    else:
        coef_space = (
            s / (s ** 2 + alpha)
        ) * projected

    coefficients = Vt.T @ coef_space

    return A @ coefficients + y_mean


def compare_pair(
    matrix_name,
    A,
    y,
    rtol,
    atol,
):
    results = {}

    decomp = {}

    for solver in (
        "custom",
        "numpy",
    ):
        U, s, Vt, info = compute_svd(
            A,
            solver=solver,
            rtol=rtol,
            atol=atol,
        )

        decomp[solver] = (
            U,
            s,
            Vt,
            info,
        )

        results[
            f"rank_{solver}"
        ] = info["rank"]

        results[
            f"rel_reconstruction_{solver}"
        ] = reconstruction_error(
            A,
            U,
            s,
            Vt,
        )

        results[
            f"orth_U_{solver}"
        ] = info["orth_U"]

        results[
            f"orth_V_{solver}"
        ] = info["orth_V"]

        results[
            f"seconds_{solver}"
        ] = info["seconds"]

        if solver == "custom":
            results["iterations"] = info[
                "iterations"
            ]
            results["converged"] = info[
                "converged"
            ]
            results["n_negative_clipped"] = (
                info["n_negative_clipped"]
            )
            results["n_small_zeroed"] = info[
                "n_small_zeroed"
            ]

    s_custom = decomp["custom"][1]
    s_numpy = decomp["numpy"][1]

    k = min(
        len(s_custom),
        len(s_numpy),
    )

    common_rank = min(
        results["rank_custom"],
        results["rank_numpy"],
    )

    results["sigma_max_abs_diff_full"] = float(
        np.max(
            np.abs(
                s_custom[:k] - s_numpy[:k]
            )
        )
    )

    results[
        "sigma_max_abs_diff_common_rank"
    ] = float(
        np.max(
            np.abs(
                s_custom[:common_rank]
                - s_numpy[:common_rank]
            )
        )
    )

    results["rank_diff"] = int(
        results["rank_custom"]
        - results["rank_numpy"]
    )

    for target_name, y_fit in [
        ("raw", y),
        ("log", np.log1p(y)),
    ]:
        for model_name, alpha in [
            ("least_squares", None),
            (
                f"ridge_{RIDGE_ALPHA:g}",
                RIDGE_ALPHA,
            ),
        ]:
            preds = {}

            for solver in (
                "custom",
                "numpy",
            ):
                U, s, Vt, _ = decomp[solver]

                rank = compute_svd_rank(
                    s,
                    rtol,
                    atol,
                    A.shape,
                )

                preds[solver] = fit_predict(
                    A,
                    U[:, :rank],
                    s[:rank],
                    Vt[:rank, :],
                    y_fit,
                    alpha,
                )

            raw = (
                target_name == "raw"
            )

            p_custom = (
                preds["custom"]
                if raw
                else np.expm1(
                    preds["custom"]
                )
            )

            p_numpy = (
                preds["numpy"]
                if raw
                else np.expm1(
                    preds["numpy"]
                )
            )

            tag = (
                f"{model_name}_pred_"
                f"{target_name}"
            )

            results[
                f"{tag}_max_abs_diff"
            ] = float(
                np.max(
                    np.abs(
                        p_custom - p_numpy
                    )
                )
            )

            results[
                f"{tag}_corr"
            ] = float(
                np.corrcoef(
                    p_custom,
                    p_numpy,
                )[0, 1]
            )

    return results


def compute_svd_rank(
    s,
    rtol,
    atol,
    shape,
):
    sigma_max = s[0]

    if rtol is None and atol is None:
        cutoff = (
            np.finfo(float).eps
            * max(shape)
            * sigma_max
        )
    else:
        cutoff = max(
            atol or 0.0,
            (rtol or 0.0) * sigma_max,
        )

    return int(
        np.count_nonzero(
            s >= cutoff
        )
    )


def main():
    # Prag comun de aplicatie, deasupra etajului de zgomot al metodei
    # Gram (A^T A): valorile singulare sub ~1e-6 * sigma_1 sunt ridicate
    # din valori proprii afectate de conditioning-ul la patrat si nu sunt
    # semnificative pentru nicio implementare.
    rtol = 1e-6
    atol = 0.0

    A_full = np.load(
        os.path.join(
            DATA,
            "A.npy"
        )
    )

    A_numeric = np.load(
        os.path.join(
            DATA,
            "A_numeric.npy"
        )
    )

    y = np.load(
        os.path.join(
            DATA,
            "y.npy"
        )
    )

    rows = []

    for matrix_name, A in [
        ("full", A_full),
        ("numeric", A_numeric),
    ]:
        print(
            f"=== {matrix_name}: "
            f"{A.shape[0]}x{A.shape[1]} ==="
        )

        res = compare_pair(
            matrix_name,
            A,
            y,
            rtol,
            atol,
        )

        res["matrix"] = matrix_name
        res["rtol"] = rtol
        res["atol"] = atol

        rows.append(res)

        print(
            "rank custom/numpy:",
            res["rank_custom"],
            "/",
            res["rank_numpy"],
            "| sigma max |diff| (rank comun):",
            f"{res['sigma_max_abs_diff_common_rank']:.3e}",
            "| recon custom/numpy:",
            f"{res['rel_reconstruction_custom']:.3e}",
            "/",
            f"{res['rel_reconstruction_numpy']:.3e}",
        )

    out = os.path.join(
        RESULTS,
        "solver_comparison.csv"
    )

    pd.DataFrame(rows).to_csv(
        out,
        index=False
    )

    print("scris:", out)


if __name__ == "__main__":
    main()
