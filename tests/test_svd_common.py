"""Teste pentru interfata comuna compute_svd (solver interschimbabil)."""

import numpy as np
import pytest

from svd_common import compute_svd, validate_matrix


def random_tall(m=150, n=25, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n))


def test_custom_matches_numpy_on_significant_spectrum():
    A = random_tall(seed=1)

    _, s_custom, _, info_custom = compute_svd(
        A,
        solver="custom",
    )

    _, s_numpy, _, info_numpy = compute_svd(
        A,
        solver="numpy",
    )

    k = min(
        info_custom["rank"],
        info_numpy["rank"],
    )

    assert k > 0

    assert np.max(
        np.abs(s_custom[:k] - s_numpy[:k])
    ) < 1e-6


def test_rank_identical_at_application_tolerance():
    # Matrice prost conditionata: la rtol=1e-6 (deasupra etajului de
    # zgomot al metodei Gram) ambele solver-e trebuie sa dea acelasi rang.
    rng = np.random.default_rng(2)

    m, n = 120, 30

    U, _ = np.linalg.qr(
        rng.standard_normal((m, n))
    )

    V, _ = np.linalg.qr(
        rng.standard_normal((n, n))
    )

    spectrum = np.logspace(
        0,
        -5,
        n,
    )

    A = (U * spectrum) @ V.T

    _, _, _, info_custom = compute_svd(
        A,
        solver="custom",
        rtol=1e-6,
    )

    _, _, _, info_numpy = compute_svd(
        A,
        solver="numpy",
        rtol=1e-6,
    )

    assert info_custom["rank"] == info_numpy["rank"]


def test_rank_cutoff_is_solver_independent():
    # Cutoff-ul post-hoc trebuie sa fie identic pentru ambele implementari.
    A = random_tall(seed=3)

    _, _, _, info_custom = compute_svd(
        A,
        solver="custom",
        rtol=1e-8,
    )

    _, _, _, info_numpy = compute_svd(
        A,
        solver="numpy",
        rtol=1e-8,
    )

    assert info_custom["cutoff"] == pytest.approx(
        info_numpy["cutoff"]
    )


def test_custom_requires_tall_matrix():
    A = random_tall(20, 50)

    with pytest.raises(
        ValueError,
        match="m >= n",
    ):
        compute_svd(
            A,
            solver="custom",
        )


def test_nan_and_inf_rejected():
    A = random_tall(seed=4)
    A[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN sau Inf",
    ):
        validate_matrix(A)

    A = random_tall(seed=5)
    A[0, 0] = np.inf

    with pytest.raises(
        ValueError,
        match="NaN sau Inf",
    ):
        validate_matrix(A)


def test_output_is_economic_format():
    A = random_tall(100, 20, seed=6)

    U, s, Vt, info = compute_svd(A)

    assert U.shape == (100, 20)
    assert s.shape == (20,)
    assert Vt.shape == (20, 20)
    assert info["s_len"] == 20


def test_singular_values_sorted_descending():
    A = random_tall(seed=7)

    _, s, _, _ = compute_svd(
        A,
        solver="custom",
    )

    assert np.all(
        np.diff(s) <= 1e-12
    )


def test_forced_deflation_stats_exposed():
    A = random_tall(seed=8)

    _, _, _, info = compute_svd(
        A,
        solver="custom",
    )

    assert "n_forced_deflations" in info
    assert "eigen_residual" in info
    assert info["eigen_residual"] < 1e-6
