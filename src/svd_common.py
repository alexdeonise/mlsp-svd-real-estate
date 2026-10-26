"""Interfața comună de SVD pentru întregul pipeline.

    U, s, Vt, info = compute_svd(A, solver="custom")

Fiecare script care are nevoie de o descompunere SVD trece prin aici, ca
solver-ul să fie interschimbabil (`custom` — implementarea principală, sau
`numpy` — linia de bază de comparație) fără să schimbe restul codului.

Convenții (uniforme pentru ambele solver-e):

- **Format de ieșire**: `U` (m×k), `s` (vector de lungime k), `Vt` (k×n),
  unde k = min(m, n) — descompunerea *economică*. Valorile singulare sunt
  întotdeauna într-un vector, nu într-o matrice diagonală.
- **`Vt`, nu `V`**: `custom_svd.SVD` întoarce `V` (n×n), aici e transpus
  imediat; NumPy întoarce direct `Vt` (k×n).
- **Ordine descrescătoare**: valorile singulare sunt sortate descrescător,
  cu aceeași permutare aplicată coloanelor lui `U` și rândurilor lui `Vt`.
- **Rang numeric comun**: `rank = #{i : s[i] > cutoff}`, cu
  `cutoff = max(atol, rtol * s[0]`. Dacă `rtol` și `atol` sunt `None`,
  se folosește convenția implicită din NumPy (`np.linalg.matrix_rank`):
  `cutoff = eps * max(m, n) * s[0]`. Atenție: convenția NumPy e legată de
  precizia float (eps ≈ 2.22e-16), nu de o toleranță de aplicație — de aceea
  `rtol`/`atol` explicit dau un alt rang, și asta e documentat în `info`.
- **Valori singulare mici / negative**: valorile negative (relicvă din
  erori numerice la extragerea valorilor proprii ale lui AᵀA) sunt tăiate la
  0 și numărate în `info["n_negative_clipped"]`; cele sub `cutoff` devin 0
  (`info["n_small_zeroed"]`). Ortogonalitatea efectivă a factorilor e
  măsurată în `info["orth_U"]` / `info["orth_V"]` — valori mari semnalează
  instabilitate numerică a solver-ului custom.
"""

import time

import numpy as np

from custom_svd import SVD

SOLVERS = (
    "custom",
    "numpy",
)


def validate_matrix(A):
    """Validează matricea de intrare; ridică ValueError cu mesaj clar."""
    A = np.asarray(
        A
    )

    if A.ndim != 2:
        raise ValueError(
            f"matricea trebuie să fie 2-D, am primit ndim={A.ndim}"
        )

    m, n = A.shape

    if m == 0 or n == 0:
        raise ValueError(
            f"matrice goală: shape={A.shape}"
        )

    if not np.all(
        np.isfinite(
            A
        )
    ):
        raise ValueError(
            "matricea conține NaN sau Inf"
        )

    return np.array(
        A,
        dtype=float,
        copy=True,
    )


def _sort_descending(
    U,
    s,
    Vt,
):
    """Sortează s descrescător, aplicând aceeași permutare lui U și Vt."""
    order = np.argsort(
        -s,
        kind="stable",
    )

    return (
        U[:, order],
        s[order],
        Vt[order, :],
    )


def compute_svd(
    A,
    solver="custom",
    rtol=None,
    atol=None,
):
    """Descompunere SVD economică cu solver interschimbabil.

    Parametri
    ---------
    A : (m, n) array
    solver : "custom" | "numpy" (implicit "custom")
        `custom` cere m >= n (SVD-ul Mariei construiește U cu n coloane
        în R^m); `numpy` funcționează pentru orice formă și e păstrat
        drept linie de bază de comparație.
    rtol, atol : float sau None
        Toleranțele criteriului de rang numeric. None + None → convenția
        NumPy. Pragul de rang e aplicat o singură dată, după rularea
        solverului, identic pentru ambele implementări — solverul custom
        primește intern doar un prag de ordinul zgomotului de mașină
        (TOL=1e-14, RTOL=0), ca decizia de rang să nu depindă de solver.

    Returnează
    ----------
    U : (m, k), s : (k,), Vt : (k, n), info : dict
        k = min(m, n); s sortat descrescător.
    """
    if solver not in SOLVERS:
        raise ValueError(
            f"solver necunoscut: {solver!r}; alege din {SOLVERS}"
        )

    A = validate_matrix(
        A
    )

    m, n = A.shape

    if solver == "custom" and m < n:
        raise ValueError(
            "solverul custom cere m >= n "
            f"(shape={A.shape}); folosește solver='numpy'"
        )

    t0 = time.time()

    backend = {}

    if solver == "numpy":
        U, s, Vt = np.linalg.svd(
            A,
            full_matrices=False,
        )

        backend = {
            "iterations": None,
            "converged": True,
        }
    else:
        U, S, V, custom_info = SVD(
            A,
            TOL=1e-14,
            RTOL=0.0,
            return_info=True,
        )

        s = np.diag(
            S
        ).copy()

        Vt = V.T

        backend = {
            "iterations": custom_info[
                "iterations"
            ],
            "converged": custom_info[
                "converged"
            ],
            "n_negative_clipped": custom_info[
                "n_negative_clipped"
            ],
            "n_small_zeroed": custom_info[
                "n_small_zeroed"
            ],
        }

    U, s, Vt = _sort_descending(
        U,
        s,
        Vt,
    )

    n_negative_clipped = int(
        np.sum(
            s < 0
        )
    )

    s = np.maximum(
        s,
        0.0,
    )

    sigma_max = (
        s[0]
        if s.size
        else 0.0
    )

    if rtol is None and atol is None:
        cutoff = (
            np.finfo(
                float
            ).eps
            * max(
                m,
                n,
            )
            * sigma_max
        )

        rtol_used = (
            np.finfo(
                float
            ).eps
            * max(
                m,
                n,
            )
        )

        atol_used = 0.0
    else:
        rtol_used = rtol or 0.0
        atol_used = atol or 0.0

        cutoff = max(
            atol_used,
            rtol_used * sigma_max,
        )

    n_small_zeroed = int(
        np.sum(
            s < cutoff
        )
    )

    s[s < cutoff] = 0.0

    rank = int(
        np.count_nonzero(
            s
        )
    )

    k = len(
        s
    )

    orth_U = float(
        np.linalg.norm(
            U.T @ U - np.eye(
                k
            )
        )
    )

    orth_V = float(
        np.linalg.norm(
            Vt @ Vt.T - np.eye(
                k
            )
        )
    )

    positive = s[
        s > 0
    ]

    info = {
        "solver": solver,
        "shape": (
            m,
            n,
        ),
        "U_shape": U.shape,
        "s_len": k,
        "Vt_shape": Vt.shape,
        "rank": rank,
        "cutoff": cutoff,
        "rtol": rtol_used,
        "atol": atol_used,
        "sigma_max": float(
            sigma_max
        ),
        "sigma_min_nonzero": (
            float(
                positive[
                    -1
                ]
            )
            if positive.size
            else None
        ),
        "n_negative_clipped": n_negative_clipped
        + backend.get(
            "n_negative_clipped",
            0,
        ),
        "n_small_zeroed": n_small_zeroed
        + backend.get(
            "n_small_zeroed",
            0,
        ),
        "orth_U": orth_U,
        "orth_V": orth_V,
        "iterations": backend[
            "iterations"
        ],
        "converged": backend[
            "converged"
        ],
        "seconds": time.time() - t0,
    }

    return U, s, Vt, info


def format_svd_info(
    info,
):
    """Reprezentare pe o linie a dicționarului info, pentru log-uri."""
    sigma_min = info[
        "sigma_min_nonzero"
    ]

    sigma_min_str = (
        f"{sigma_min:.3e}"
        if sigma_min is not None
        else "None"
    )

    parts = [
        f"solver={info['solver']}",
        f"shape={info['shape'][0]}x{info['shape'][1]}",
        f"rank={info['rank']}",
        f"cutoff={info['cutoff']:.3e}",
        f"rtol={info['rtol']:.3e}",
        f"atol={info['atol']:.3e}",
        f"sigma_max={info['sigma_max']:.3e}",
        f"sigma_min={sigma_min_str}",
        f"orth_U={info['orth_U']:.3e}",
        f"orth_V={info['orth_V']:.3e}",
    ]

    if info[
        "iterations"
    ] is not None:
        parts.append(
            f"iterations={info['iterations']}"
        )

        parts.append(
            f"converged={info['converged']}"
        )

    if info[
        "n_negative_clipped"
    ]:
        parts.append(
            f"neg_clipped={info['n_negative_clipped']}"
        )

    if info[
        "n_small_zeroed"
    ]:
        parts.append(
            f"small_zeroed={info['n_small_zeroed']}"
        )

    parts.append(
        f"seconds={info['seconds']:.2f}"
    )

    return "[svd] " + " ".join(
        parts
    )
