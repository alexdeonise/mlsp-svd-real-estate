import math
import numpy as np


def _norm(x):
    return math.sqrt(float(sum(xi * xi for xi in x)))


def _sign(x):
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0


def _outer(v, w):
    v = np.asarray(v)
    w = np.asarray(w)

    m = len(v)
    n = len(w)

    result = np.zeros((m, n))

    for i in range(m):
        vi = float(v[i])

        for j in range(n):
            result[i, j] = vi * float(w[j])

    return result


def _eye(n):
    I = np.zeros((n, n))

    for i in range(n):
        I[i, i] = 1.0

    return I


def _zeros(m, n):
    return np.zeros((m, n))


def _diag(A):
    n = min(A.shape[0], A.shape[1])

    return np.array([
        float(A[i, i])
        for i in range(n)
    ])


def _diag_matrix(d):
    n = len(d)

    A = np.zeros((n, n))

    for i in range(n):
        A[i, i] = d[i]

    return A


def _tril(A, k=0):
    m, n = A.shape

    R = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            if j <= i + k:
                R[i, j] = A[i, j]

    return R


def _triu(A, k=0):
    m, n = A.shape

    R = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            if j >= i + k:
                R[i, j] = A[i, j]

    return R


def _argsort_desc(arr):
    indexed = [
        (float(arr[i]), i)
        for i in range(len(arr))
    ]

    indexed.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        idx
        for _, idx in indexed
    ]


def _sqrt_arr(arr):
    return np.array([
        math.sqrt(x) if x > 0 else 0.0
        for x in arr
    ])


def Tridiag_Householder(A):
    n = A.shape[0]

    T = np.copy(A).astype(float)

    Q = _eye(n)

    for k in range(n - 2):
        x = T[k + 1:, k].copy()

        norm_x = _norm(x)

        if abs(norm_x) < 1e-14:
            continue

        sgn = (
            _sign(x[0])
            if abs(x[0]) > 1e-14
            else 1.0
        )

        x[0] += sgn * norm_x

        v = x / _norm(x)

        T[k + 1:, :] -= 2.0 * _outer(
            v,
            v @ T[k + 1:, :]
        )

        T[:, k + 1:] -= 2.0 * _outer(
            T[:, k + 1:] @ v,
            v
        )

        Q[:, k + 1:] -= 2.0 * _outer(
            Q[:, k + 1:] @ v,
            v
        )

    T = _tril(
        _triu(T, -1),
        1
    )

    T = (T + T.T) / 2.0

    return Q, T


def _qr_step_tridiag_explicit(T_block):
    n = T_block.shape[0]

    T_shifted = T_block.copy()

    delta = (
        T_shifted[n - 2, n - 2]
        -
        T_shifted[n - 1, n - 1]
    ) / 2.0

    sign = (
        _sign(delta)
        if abs(delta) > 1e-14
        else 1.0
    )

    mu = (
        T_shifted[n - 1, n - 1]
        -
        T_shifted[n - 2, n - 1] ** 2
        /
        (
            delta
            +
            sign
            *
            math.sqrt(
                delta ** 2
                +
                T_shifted[n - 2, n - 1] ** 2
            )
        )
    )

    T_shifted -= mu * _eye(n)

    rotations = []

    for k in range(n - 1):
        a = T_shifted[k, k]
        b = T_shifted[k + 1, k]

        r = math.hypot(a, b)

        if r < 1e-14:
            rotations.append(
                (1.0, 0.0)
            )
            continue

        c = a / r
        s = b / r

        row_k = T_shifted[k, :].copy()
        row_k1 = T_shifted[k + 1, :].copy()

        T_shifted[k, :] = (
            c * row_k
            +
            s * row_k1
        )

        T_shifted[k + 1, :] = (
            -s * row_k
            +
            c * row_k1
        )

        rotations.append(
            (c, s)
        )

    for k in range(n - 1):
        c, s = rotations[k]

        col_k = T_shifted[:, k].copy()
        col_k1 = T_shifted[:, k + 1].copy()

        T_shifted[:, k] = (
            c * col_k
            +
            s * col_k1
        )

        T_shifted[:, k + 1] = (
            -s * col_k
            +
            c * col_k1
        )

    T_new = (
        T_shifted
        +
        mu * _eye(n)
    )

    T_new = _tril(
        _triu(T_new, -1),
        1
    )

    T_new = (
        T_new
        +
        T_new.T
    ) / 2.0

    return T_new, rotations


def QR_iteration(A, Q, TOL=1e-6):
    T = Q.T @ A @ Q

    n = A.shape[0]

    V = Q.copy()

    active_end = n

    total_iters = 0

    max_total_iters = 5000

    # Deflatii fortate: blocuri care nu au ajuns sub TOL in cele 10
    # iteratii ale buclei interne si au fost dezmembrate prin zeroarea
    # fortata a elementului subdiagonal. Sunt numarate separat, cu
    # reziduul lor, ca sa nu fie confundate cu deflatiile prin convergenta.
    n_forced = 0

    max_forced_residual = 0.0

    while (
        active_end > 1
        and
        total_iters < max_total_iters
    ):
        while (
            active_end > 1
            and
            abs(
                T[
                    active_end - 1,
                    active_end - 2
                ]
            ) < TOL
        ):
            active_end -= 1

        if active_end <= 1:
            break

        block_iters = 0

        while (
            abs(
                T[
                    active_end - 1,
                    active_end - 2
                ]
            ) >= TOL
            and
            block_iters < 10
        ):
            T_block = T[
                :active_end,
                :active_end
            ].copy()

            T_block_new, rots = (
                _qr_step_tridiag_explicit(
                    T_block
                )
            )

            T[
                :active_end,
                :active_end
            ] = T_block_new

            for k, (c, s) in enumerate(rots):
                v_k = V[:, k].copy()
                v_k1 = V[:, k + 1].copy()

                V[:, k] = (
                    c * v_k
                    +
                    s * v_k1
                )

                V[:, k + 1] = (
                    -s * v_k
                    +
                    c * v_k1
                )

            total_iters += 1

            block_iters += 1

        if active_end > 1:
            residual = abs(
                T[
                    active_end - 1,
                    active_end - 2
                ]
            )

            if residual >= TOL:
                n_forced += 1

                max_forced_residual = max(
                    max_forced_residual,
                    residual,
                )

            T[
                active_end - 1,
                active_end - 2
            ] = 0.0

            T[
                active_end - 2,
                active_end - 1
            ] = 0.0

            active_end -= 1

    D = _diag_matrix(
        _diag(T)
    )

    converged = active_end <= 1

    # Certificat de convergenta: (V, D) aproximeaza matricea originala A
    # (in bazele ei, nu in baza tridiagonala — V acumuleaza si pe Q0).
    # Mare => deflatii fortate cu reziduuri mari sau epuizarea buclei
    # externe — ambele vizibile in stats.
    R = A - V @ D @ V.T

    denom = np.linalg.norm(A)

    eigen_residual = (
        float(np.linalg.norm(R) / denom)
        if denom > 0
        else float(np.linalg.norm(R))
    )

    stats = {
        "n_forced_deflations": n_forced,
        "max_forced_deflation_residual": max_forced_residual,
        "eigen_residual": eigen_residual,
    }

    return D, V, total_iters, converged, stats


def _complete_orthonormal_basis(U, constructed, missing, TOL=1e-12):
    if len(missing) == 0:
        return U

    m = U.shape[0]

    basis_indices = list(constructed)

    candidate_index = 0

    for target in missing:
        found = False

        while candidate_index < m:
            u = np.zeros(m)

            u[candidate_index] = 1.0

            candidate_index += 1

            if basis_indices:
                Q = U[:, basis_indices]

                u = u - Q @ (Q.T @ u)
                u = u - Q @ (Q.T @ u)

            norm_u = np.linalg.norm(u)

            if norm_u > TOL:
                U[:, target] = u / norm_u

                basis_indices.append(target)

                found = True

                break

        if not found:
            raise RuntimeError(
                "Nu s-a putut completa baza ortonormala pentru U."
            )

    return U


def SVD(A, TOL=1e-14, RTOL=1e-6, return_info=False):
    A = np.copy(A).astype(float)

    m = A.shape[0]
    n = A.shape[1]

    B = A.T @ A

    Q0, T0 = Tridiag_Householder(B)

    D, V, qr_iters, qr_converged, qr_stats = QR_iteration(
        B,
        Q0
    )

    val = _diag(D)

    indici = _argsort_desc(val)

    val = val[indici]

    V = V[:, indici]

    n_negative = int(
        np.sum(
            val < 0
        )
    )

    val[val < 0] = 0.0

    sigma = _sqrt_arr(val)

    if len(sigma) > 0:
        sigma_max = sigma[0]
    else:
        sigma_max = 0.0

    cutoff = max(
        TOL,
        RTOL * sigma_max
    )

    n_small = int(
        np.sum(
            sigma < cutoff
        )
    )

    sigma[
        sigma < cutoff
    ] = 0.0

    S = _zeros(
        n,
        n
    )

    for i in range(n):
        S[i, i] = sigma[i]

    U = _zeros(
        m,
        n
    )

    valid = [
        i
        for i in range(n)
        if sigma[i] > cutoff
    ]

    missing = [
        i
        for i in range(n)
        if sigma[i] <= cutoff
    ]

    constructed = []

    for i in valid:
        u = (
            A @ V[:, i]
        ) / sigma[i]

        if constructed:
            Q = U[:, constructed]

            u = u - Q @ (Q.T @ u)
            u = u - Q @ (Q.T @ u)

        norm_u = np.linalg.norm(u)

        if norm_u > TOL:
            U[:, i] = (
                u / norm_u
            )

            constructed.append(i)
        else:
            missing.append(i)

    missing = sorted(
        list(set(missing) - set(constructed))
    )

    U = _complete_orthonormal_basis(
        U,
        constructed,
        missing
    )

    if return_info:
        info = {
            "iterations": qr_iters,
            "converged": bool(
                qr_converged
            ),
            "qr_tol": 1e-6,
            "cutoff": cutoff,
            "rtol": RTOL,
            "atol": TOL,
            "n_negative_clipped": n_negative,
            "n_small_zeroed": n_small,
            "n_constructed_u": len(
                constructed
            ),
            "n_forced_deflations": qr_stats[
                "n_forced_deflations"
            ],
            "max_forced_deflation_residual": qr_stats[
                "max_forced_deflation_residual"
            ],
            "eigen_residual": qr_stats[
                "eigen_residual"
            ],
        }

        return U, S, V, info

    return U, S, V