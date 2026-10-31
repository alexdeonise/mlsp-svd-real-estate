"""Consistenta artefactelor cu datasetul real.

Aceste teste nu ruleaza in CI (datele *.npy/*.csv sunt in .gitignore);
se ruleaza local, unde artefactele din results/ trebuie sa corespunda
calculelor proaspete. Daca fisierul de date lipseste, testele sunt
sarite explicit.
"""

import os

import numpy as np
import pandas as pd
import pytest

from regression import make_split
from svd_common import compute_svd

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

A_NUMERIC = os.path.join(DATA, "A_numeric.npy")
SPLIT_CSV = os.path.join(RESULTS, "regression_split.csv")
QR_STATS_CSV = os.path.join(RESULTS, "svd_qr_stats.csv")
TT_METRICS_CSV = os.path.join(RESULTS, "regression_tt_metrics.csv")

needs_numeric = pytest.mark.skipif(
    not (
        os.path.exists(A_NUMERIC)
        and os.path.exists(QR_STATS_CSV)
    ),
    reason="data/A_numeric.npy sau svd_qr_stats.csv nu exista local",
)

needs_artifacts = pytest.mark.skipif(
    not (
        os.path.exists(SPLIT_CSV)
        and os.path.exists(QR_STATS_CSV)
    ),
    reason="artefactele results/ lipsesc",
)


@needs_numeric
def test_numeric_svd_matches_qr_stats_artifact():
    """SVD proaspat pe A_numeric trebuie sa coincida cu svd_qr_stats.csv."""
    A = np.load(A_NUMERIC)

    _, s, _, info = compute_svd(
        A,
        solver="custom",
        rtol=1e-6,
    )

    stats = pd.read_csv(QR_STATS_CSV)

    row = stats[
        (stats["matrix"] == "numeric")
        & (stats["solver"] == "custom")
    ]

    assert len(row) == 1, "lipsesc randurile numeric/custom din svd_qr_stats.csv"

    row = row.iloc[0]

    assert int(row["rank"]) == info["rank"]
    assert row["sigma_max"] == pytest.approx(
        info["sigma_max"],
        rel=1e-9,
    )

    if not np.isnan(row["eigen_residual"]):
        assert row["eigen_residual"] == pytest.approx(
            info["eigen_residual"],
            rel=1e-6,
        )

    assert bool(row["converged"])


@needs_artifacts
def test_regression_split_reproducible():
    """regression_split.csv trebuie sa fie reprodus de make_split(seed=42)."""
    saved = pd.read_csv(SPLIT_CSV)

    n = len(saved)

    train_idx, val_idx, test_idx = make_split(
        n,
        seed=int(saved["seed"].iloc[0]),
    )

    reconstructed = np.full(
        n,
        "",
        dtype=object,
    )

    reconstructed[train_idx] = "train"
    reconstructed[val_idx] = "val"
    reconstructed[test_idx] = "test"

    assert (reconstructed == saved["split"].to_numpy()).all()


@needs_artifacts
def test_qr_stats_schema_and_convergence():
    """svd_qr_stats.csv trebuie sa aiba coloanele de convergenta populate."""
    stats = pd.read_csv(QR_STATS_CSV)

    for column in [
        "matrix",
        "solver",
        "rank",
        "cutoff",
        "converged",
        "eigen_residual",
        "n_forced_deflations",
    ]:
        assert column in stats.columns, f"lipseste coloana {column}"

    custom = stats[stats["solver"] == "custom"]

    assert (custom["converged"] == True).all()  # noqa: E712
    assert (custom["eigen_residual"] < 1e-6).all()


def test_tt_metrics_schema():
    """Tabelul principal de metrici: coloanele de protocol M2 trebuie sa existe."""
    if not os.path.exists(TT_METRICS_CSV):
        pytest.skip("regression_tt_metrics.csv lipseste")

    df = pd.read_csv(TT_METRICS_CSV)

    for column in [
        "solver",
        "target",
        "model",
        "alpha",
        "fit_split",
        "split",
        "rmse",
        "mae",
        "r2",
        "selected_alpha",
    ]:
        assert column in df.columns, f"lipseste coloana {column}"

    # Selectia alpha: exact un rand selectat pe (solver, target), doar pe test.
    selected = df[df["selected_alpha"] == True]  # noqa: E712

    grouped = selected.groupby(
        ["solver", "target"]
    )

    for _, group in grouped:
        assert (group["split"] == "test").all()
        assert len(group) == 1
