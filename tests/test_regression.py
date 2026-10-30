"""Teste pentru componentele regresiei train/test.

Pipeline-ul complet (cu SVD pe matricea reala) e acoperit de scripturile
de rulat; aici testam unitar proprietatile metodologice: reproductibilitatea
splitului, exclusivitatea standardizarii pe train, selectia lui alpha pe
validare si corectitudinea metricilor.
"""

import numpy as np
import pytest

from regression import (
    compute_metrics,
    fit_coefficients,
    make_split,
    select_alpha,
    standardize_on_train,
)


def test_make_split_reproducible():
    n = 1000

    t1, v1, s1 = make_split(n, seed=42)
    t2, v2, s2 = make_split(n, seed=42)

    assert np.array_equal(t1, t2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(s1, s2)


def test_make_split_disjoint_and_complete():
    n = 1234

    train, val, test = make_split(n, seed=7)

    all_idx = np.concatenate(
        [train, val, test]
    )

    assert len(all_idx) == n
    assert len(np.unique(all_idx)) == n

    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)


def test_make_split_different_seeds_differ():
    n = 500

    _, _, test_a = make_split(n, seed=1)
    _, _, test_b = make_split(n, seed=2)

    assert not np.array_equal(
        np.sort(test_a),
        np.sort(test_b),
    )


def test_standardize_uses_train_statistics_only():
    rng = np.random.default_rng(0)

    n = 200
    X = rng.uniform(
        1.0,
        2.0,
        size=(n, 3),
    )

    train_idx = np.arange(100)
    val_idx = np.arange(100, 150)
    test_idx = np.arange(150, 200)

    Xs = standardize_on_train(
        X,
        train_idx,
    )

    train_std = Xs[train_idx]

    assert np.allclose(
        train_std.mean(axis=0),
        0.0,
        atol=1e-12,
    )

    assert np.allclose(
        train_std.std(axis=0),
        1.0,
        atol=1e-12,
    )

    # Modificarea randurilor de test NU trebuie sa schimbe transformarea.
    X_perturbed = X.copy()

    X_perturbed[test_idx] *= 1e6
    X_perturbed[val_idx] += 5.0

    Xs_perturbed = standardize_on_train(
        X_perturbed,
        train_idx,
    )

    assert np.array_equal(
        Xs[train_idx],
        Xs_perturbed[train_idx],
    )


def test_standardize_zero_variance_column_no_nan():
    X = np.ones((50, 2))
    X[:, 1] = np.linspace(1.0, 2.0, 50)

    Xs = standardize_on_train(
        X,
        np.arange(25),
    )

    assert np.all(np.isfinite(Xs))
    assert np.all(Xs[:, 0] == 0.0)


def test_least_squares_recovers_exact_coefficients():
    rng = np.random.default_rng(1)

    m, n = 80, 10

    X = rng.standard_normal((m, n))

    # Coloane centrate: modelul cu y centrat (conventia din regression.py)
    # recupereaza exact coeficientii.
    X = X - X.mean(axis=0)

    coef_true = rng.standard_normal(n)

    y = X @ coef_true + 5.0

    U, s, Vt = np.linalg.svd(
        X,
        full_matrices=False,
    )

    y_mean = np.mean(y)
    coef = fit_coefficients(
        U,
        s,
        Vt,
        y - y_mean,
    )

    assert np.allclose(
        coef,
        coef_true,
        atol=1e-8,
    )


def test_ridge_shrinks_coefficients():
    rng = np.random.default_rng(2)

    m, n = 60, 15

    X = rng.standard_normal((m, n))

    coef_true = rng.standard_normal(n)
    y = X @ coef_true + 0.5 * rng.standard_normal(m)

    U, s, Vt = np.linalg.svd(
        X,
        full_matrices=False,
    )

    y_mean = np.mean(y)
    y_centered = y - y_mean

    coef_ls = fit_coefficients(
        U,
        s,
        Vt,
        y_centered,
    )

    coef_ridge = fit_coefficients(
        U,
        s,
        Vt,
        y_centered,
        alpha=100.0,
    )

    assert np.linalg.norm(
        coef_ridge
    ) < np.linalg.norm(
        coef_ls
    )


def test_select_alpha_uses_validation_metrics():
    # Selectia se face exclusiv pe metricile de validare primite ca
    # argument — functia nu primeste nicio metrica de test.
    candidates = [
        ("ridge_alpha_1", 1.0, {"rmse": 10.0}),
        ("ridge_alpha_10", 10.0, {"rmse": 7.5}),
        ("ridge_alpha_100", 100.0, {"rmse": 9.0}),
    ]

    name, alpha, metrics = select_alpha(candidates)

    assert name == "ridge_alpha_10"
    assert alpha == 10.0
    assert metrics["rmse"] == 7.5


def test_select_alpha_rejects_empty():
    with pytest.raises(ValueError):
        select_alpha([])


def test_compute_metrics_perfect_prediction():
    y = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    m = compute_metrics(y, y.copy())

    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == pytest.approx(1.0)


def test_compute_metrics_worse_than_mean_negative_r2():
    y = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    # Predictii cu eroare mai mare decat media constanta.
    predictions = np.array(
        [-10.0, 20.0, -30.0, 40.0]
    )

    m = compute_metrics(y, predictions)

    assert m["r2"] < 0.0
