"""Teste pentru SVD-ul custom: convergență QR, cazuri numerice dificile."""

import numpy as np
import pytest

from custom_svd import QR_iteration, SVD, Tridiag_Householder


def random_spd(n, seed=0, spectrum=None):
    """Matrice simetrică pozitiv definită cu spectru controlat."""
    rng = np.random.default_rng(seed)

    Q, _ = np.linalg.qr(
        rng.standard_normal((n, n))
    )

    if spectrum is None:
        spectrum = np.sort(
            rng.uniform(0.1, 100.0, size=n)
        )[::-1]

    return (
        Q * spectrum
    ) @ Q.T, np.asarray(spectrum)


def test_diagonal_matrix_eigenvalues_exact():
    n = 8
    lam = np.array(
        [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    )

    A = np.diag(lam)

    U, S, V, info = SVD(
        A,
        TOL=1e-14,
        RTOL=0.0,
        return_info=True,
    )

    sigma = np.diag(S)[:n]

    assert info["converged"]
    assert info["n_forced_deflations"] == 0
    assert info["eigen_residual"] < 1e-12
    assert np.max(np.abs(sigma - lam)) < 1e-8


def test_no_forced_deflation_on_separated_spectrum():
    n = 30

    A, spectrum = random_spd(
        n,
        seed=1,
        spectrum=np.linspace(
            100.0,
            1.0,
            n,
        ),
    )

    _, _, _, info = SVD(
        A,
        TOL=1e-14,
        RTOL=0.0,
        return_info=True,
    )

    assert info["converged"]
    assert info["n_forced_deflations"] == 0
    assert info["eigen_residual"] < 1e-10


def test_repeated_eigenvalues_converge():
    # Spectru cu grupe stranse (aproape degenerate): aici deflatiile
    # fortate apar cel mai des; trebuie sa fie raportate, nu ascunse.
    n = 24

    spectrum = np.repeat(
        [100.0, 10.0, 1.0, 0.1],
        n // 4,
    )

    A, _ = random_spd(
        n,
        seed=2,
        spectrum=spectrum,
    )

    U, S, V, info = SVD(
        A,
        TOL=1e-14,
        RTOL=0.0,
        return_info=True,
    )

    assert info["converged"]

    # Certificatul trebuie sa ramana mic indiferent cum s-au facut deflatiile.
    assert info["eigen_residual"] < 1e-6

    # Ortogonalitatea factorilor nu se degradeaza.
    assert np.linalg.norm(
        U.T @ U - np.eye(n)
    ) < 1e-8

    assert np.linalg.norm(
        V.T @ V - np.eye(n)
    ) < 1e-8


def test_stats_keys_present_and_consistent():
    n = 20

    A, _ = random_spd(n, seed=3)

    _, _, _, info = SVD(
        A,
        TOL=1e-14,
        RTOL=0.0,
        return_info=True,
    )

    assert info["n_forced_deflations"] >= 0
    assert info["max_forced_deflation_residual"] >= 0.0
    assert 0.0 <= info["eigen_residual"] < 1e-6

    if info["n_forced_deflations"] == 0:
        assert info["max_forced_deflation_residual"] == 0.0


def test_small_negative_eigenvalues_clipped_not_nan():
    # Coloane aproape liniar dependente => B aproape singulara =>
    # valori proprii mici, unele negative din rotunjire.
    rng = np.random.default_rng(4)

    base = rng.standard_normal(60)
    A = np.column_stack(
        [
            base,
            base + 1e-9,
            rng.standard_normal(60),
        ]
    )

    U, S, V, info = SVD(
        A,
        TOL=1e-14,
        RTOL=0.0,
        return_info=True,
    )

    sigma = np.diag(S)

    assert np.all(np.isfinite(sigma))
    assert np.all(sigma >= 0.0)
    assert info["n_negative_clipped"] >= 0


def test_reconstruction_random_tall_matrix():
    rng = np.random.default_rng(5)

    m, n = 120, 20
    A = rng.standard_normal((m, n))

    U, S, V, info = SVD(
        A,
        TOL=1e-14,
        RTOL=0.0,
        return_info=True,
    )

    sigma = np.diag(S)

    Ak = (U * sigma) @ V.T

    rel = (
        np.linalg.norm(A - Ak)
        / np.linalg.norm(A)
    )

    assert rel < 1e-8
    assert info["eigen_residual"] < 1e-8


def test_qr_iteration_max_iterations_guard():
    # O matrice care nu converge in 5000 de iteratii nu exista in practica;
    # verificam doar ca bucla se termina si raporteaza corect epuizarea.
    n = 6

    A, _ = random_spd(n, seed=6)

    Q0, T0 = Tridiag_Householder(A)

    D, V, iters, converged, stats = QR_iteration(
        A,
        Q0,
        TOL=1e-300,
    )

    assert iters <= 5000

    if not converged:
        assert stats["eigen_residual"] > 0


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_sigma_matches_numpy_over_significant_spectrum(seed):
    rng = np.random.default_rng(seed)

    m, n = 200, 40
    A = rng.standard_normal((m, n))

    _, S, _ = SVD(
        A,
        TOL=1e-14,
        RTOL=0.0,
    )

    s_custom = np.diag(S)

    s_numpy = np.linalg.svd(
        A,
        compute_uv=False,
    )

    k = min(
        np.linalg.matrix_rank(A),
        n,
    )

    diff = np.max(
        np.abs(s_custom[:k] - s_numpy[:k])
    )

    assert diff < 1e-6
