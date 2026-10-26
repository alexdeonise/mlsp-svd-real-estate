"""Regresia pretului de vanzare cu evaluare corecta pe train/test.

Pipeline (M2):
  1. Design matrix brut (one-hot, nestandardizat) reconstruit din CSV.
  2. Split reproducibil 80/20 (seed fix, salvat in results/regression_split.csv).
  3. Standardizare fit EXCLUSIV pe train (media/deviatia de pe train,
     aplicate si pe val/test) — niciun leakage din test in preprocessing.
  4. SVD pe matricea de train (solver interschimbabil, custom implicit),
     cu acelasi prag de rang pentru ambele solver-e.
  5. Subsplit train -> train/validation pentru alegerea lui alpha la Ridge.
  6. Least squares + Ridge: coeficienti fitati pe train, predictii pe
     train si test; metrici (RMSE, MAE, R²) raportate separat pe fiecare
     split. Pentru --log-target predictiile sunt convertite inapoi in USD
     (expm1) INAINTE de calculul metricelor in dolari.

Scrie:
  results/regression_tt_metrics.csv            tabelul principal de metrici
  results/regression_split.csv                 indicii splitului + seed
  results/log_target_diagnostics.csv           diagnostic pentru R² negativ
  results/{model}_top50_{under/overvalued}{_log}_{solver}.csv
  results/regression_neighborhood_concentration{_log}_{solver}.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from build_matrix import CAT, NUM, load_clean
from svd_common import compute_svd, format_svd_info

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

SEED = 42
TEST_FRACTION = 0.20
VAL_FRACTION_OF_TRAIN = 0.20


def load_design(csv_path):
    """Design matrix brut (nestandardizat) + preturi + dataframe curatat."""
    df = load_clean(csv_path)

    X = pd.get_dummies(
        df[NUM + CAT],
        columns=CAT,
        drop_first=False,
    ).astype(float)

    y = df["SALE PRICE"].to_numpy(float)

    return X.to_numpy(), y, df


def make_split(n, seed=SEED):
    """Split reproducibil 80/20 + subsplit 80/20 al partii de train.

    Intoarce (train_idx, val_idx, test_idx), disjuncte, reuniunea = toate
    indicii. Acelasi seed reproduce exact acelasi split.
    """
    rng = np.random.default_rng(seed)

    perm = rng.permutation(n)

    n_test = int(round(TEST_FRACTION * n))
    test_idx = perm[:n_test]
    rest = perm[n_test:]

    n_val = int(round(VAL_FRACTION_OF_TRAIN * len(rest)))
    val_idx = rest[:n_val]
    train_idx = rest[n_val:]

    return (
        np.sort(train_idx),
        np.sort(val_idx),
        np.sort(test_idx),
    )


def save_split(train_idx, val_idx, test_idx, seed):
    split = np.full(
        len(train_idx) + len(val_idx) + len(test_idx),
        "",
        dtype=object,
    )

    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"

    pd.DataFrame({
        "row_id": np.arange(len(split)),
        "split": split,
        "seed": seed,
    }).to_csv(
        os.path.join(RESULTS, "regression_split.csv"),
        index=False,
    )


def standardize_on_train(X, train_idx):
    """Standardizeaza coloanele folosind doar randurile de train.

    Coloanele cu deviatie zero pe train devin 0 (scalam cu 1 in loc de 0,
    la fel ca in build_matrix.standardize).
    """
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    sd[sd == 0.0] = 1.0

    return (X - mu) / sd


def fit_coefficients(U, s, Vt, y_centered, alpha=None):
    """Coeficientii least squares (alpha=None) sau Ridge in spatiul SVD."""
    projected = U.T @ y_centered

    if alpha is None:
        coef_space = projected / s
    else:
        coef_space = (s / (s ** 2 + alpha)) * projected

    return Vt.T @ coef_space


def predict_split(A_split, coefficients, y_mean):
    return A_split @ coefficients + y_mean


def compute_metrics(y, predictions):
    residuals = y - predictions

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r2 = 1.0 - ss_res / ss_tot

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        "ss_res": ss_res,
        "ss_tot": ss_tot,
    }


def make_residual_table(df, y, predictions, split_mask, split_name, model_name):
    y_split = y[split_mask]

    residuals = y_split - predictions

    result = pd.DataFrame({
        "row_id": np.arange(len(df))[split_mask],
        "split": split_name,
        "model": model_name,
        "sale_price": y_split,
        "predicted_price": predictions,
        "residual": residuals,
        "absolute_residual": np.abs(residuals),
    })

    metadata_columns = [
        "BOROUGH",
        "NEIGHBORHOOD",
        "BUILDING CLASS CATEGORY",
        "ADDRESS",
        "ZIP CODE",
    ]

    for column in metadata_columns:
        if column in df.columns:
            result[column] = df.iloc[
                result["row_id"]
            ][column].to_numpy()

    return result.sort_values(
        "absolute_residual",
        ascending=False,
    ).head(50)


def save_top_residuals(residual_table, model_name, suffix, solver):
    undervalued = residual_table.sort_values(
        "residual",
        ascending=True,
    ).head(50)

    overvalued = residual_table.sort_values(
        "residual",
        ascending=False,
    ).head(50)

    undervalued.to_csv(
        os.path.join(
            RESULTS,
            f"{model_name}_top50_undervalued{suffix}_{solver}.csv",
        ),
        index=False,
    )

    overvalued.to_csv(
        os.path.join(
            RESULTS,
            f"{model_name}_top50_overvalued{suffix}_{solver}.csv",
        ),
        index=False,
    )

    return undervalued, overvalued


def neighborhood_summary(undervalued, overvalued, model_name, solver):
    rows = []

    for group_name, group in [
        ("undervalued", undervalued),
        ("overvalued", overvalued),
    ]:
        counts = group["NEIGHBORHOOD"].value_counts()

        for neighborhood, count in counts.items():
            rows.append({
                "model": model_name,
                "solver": solver,
                "group": group_name,
                "neighborhood": neighborhood,
                "count": int(count),
                "share": count / 50.0,
            })

    return rows


def log_diagnostics(model_name, solver, y_test, predictions_usd):
    """Statistici pentru investigatia R² negativ pe log(1+pret)."""
    residuals = y_test - predictions_usd

    order = np.argsort(-np.abs(residuals))

    top5_share = float(
        np.sum(residuals[order[:5]] ** 2)
        / np.sum(residuals ** 2)
    ) if np.sum(residuals ** 2) > 0 else 0.0

    return {
        "model": model_name,
        "solver": solver,
        "n_test": len(y_test),
        "max_pred_usd": float(np.max(predictions_usd)),
        "median_pred_usd": float(np.median(predictions_usd)),
        "max_actual_usd": float(np.max(y_test)),
        "n_pred_gt_1e8": int(np.sum(predictions_usd > 1e8)),
        "n_pred_gt_1e9": int(np.sum(predictions_usd > 1e9)),
        "n_pred_negative": int(np.sum(predictions_usd < 0)),
        "top5_residuals_share_of_ss_res": top5_share,
        "ss_res": float(np.sum(residuals ** 2)),
        "ss_tot": float(np.sum((y_test - np.mean(y_test)) ** 2)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("csv")

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[
            100.0,
            1000.0,
            10000.0,
            100000.0,
        ],
    )

    parser.add_argument(
        "--solver",
        choices=["custom", "numpy"],
        default="custom",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--log-target",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    X_raw, y, df = load_design(args.csv)

    print("X:", X_raw.shape)
    print("solver:", args.solver)
    print("log target:", args.log_target)
    print("seed:", args.seed)

    train_idx, val_idx, test_idx = make_split(
        len(df),
        seed=args.seed,
    )

    save_split(train_idx, val_idx, test_idx, args.seed)

    print(
        "split: train",
        len(train_idx),
        "| val",
        len(val_idx),
        "| test",
        len(test_idx),
    )

    X = standardize_on_train(X_raw, train_idx)

    A_train = X[train_idx]
    A_val = X[val_idx]
    A_test = X[test_idx]

    U_full, s_full, Vt_full, svd_info = compute_svd(
        A_train,
        solver=args.solver,
        rtol=args.rtol,
        atol=args.atol,
    )

    print(format_svd_info(svd_info))

    rank = svd_info["rank"]

    U = U_full[:, :rank]
    s = s_full[:rank]
    Vt = Vt_full[:rank, :]

    print("numerical rank (train):", rank)

    if args.log_target:
        y_fit_all = np.log1p(y)
        suffix = "_log"
    else:
        y_fit_all = y.copy()
        suffix = ""

    y_mean = np.mean(y_fit_all[train_idx])
    y_centered_train = y_fit_all[train_idx] - y_mean

    metric_rows = []
    neighborhood_rows = []
    diagnostics_rows = []
    top_residual_tables = {}

    model_specs = [("least_squares", None)]
    model_specs.extend(
        (f"ridge_alpha_{alpha:g}", alpha)
        for alpha in args.alphas
    )

    for model_name, alpha in model_specs:
        coefficients = fit_coefficients(
            U,
            s,
            Vt,
            y_centered_train,
            alpha,
        )

        predictions_fit = {
            "train": predict_split(
                A_train,
                coefficients,
                y_mean,
            ),
            "val": predict_split(
                A_val,
                coefficients,
                y_mean,
            ),
            "test": predict_split(
                A_test,
                coefficients,
                y_mean,
            ),
        }

        if args.log_target:
            predictions = {
                split: np.expm1(p)
                for split, p in predictions_fit.items()
            }
        else:
            predictions = predictions_fit

        for split_name in ("train", "test"):
            m = compute_metrics(
                y[
                    train_idx
                    if split_name == "train"
                    else test_idx
                ],
                predictions[split_name],
            )

            metric_rows.append({
                "solver": args.solver,
                "target": "log" if args.log_target else "raw",
                "model": model_name,
                "alpha": "" if alpha is None else alpha,
                "split": split_name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "rank": rank,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "seed": args.seed,
            })

        m_test = compute_metrics(
            y[test_idx],
            predictions["test"],
        )

        m_val = compute_metrics(
            y[val_idx],
            predictions["val"],
        )

        print(
            model_name + ":",
            f"val RMSE={m_val['rmse']:.3f}",
            f"| test RMSE={m_test['rmse']:.3f}",
            f"MAE={m_test['mae']:.3f}",
            f"R2={m_test['r2']:.6f}",
        )

        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_idx] = True

        residual_table = make_residual_table(
            df,
            y,
            predictions["test"],
            test_mask,
            "test",
            model_name,
        )

        top_residual_tables[model_name] = residual_table

        undervalued, overvalued = save_top_residuals(
            residual_table,
            model_name,
            suffix,
            args.solver,
        )

        neighborhood_rows.extend(
            neighborhood_summary(
                undervalued,
                overvalued,
                model_name,
                args.solver,
            )
        )

        if args.log_target:
            diagnostics_rows.append(
                log_diagnostics(
                    model_name,
                    args.solver,
                    y[test_idx],
                    predictions["test"],
                )
            )

            if (
                model_name == "least_squares"
                and args.solver == "custom"
            ):
                pd.DataFrame({
                    "row_id": test_idx,
                    "actual_usd": y[test_idx],
                    "predicted_usd": predictions["test"],
                }).to_csv(
                    os.path.join(
                        RESULTS,
                        "log_ls_test_predictions.csv",
                    ),
                    index=False,
                )

    metrics_df = pd.DataFrame(metric_rows)

    metrics_df["selected_alpha"] = False

    for split_target in ("train", "test"):
        subset = metrics_df[
            (metrics_df["split"] == split_target)
            & (metrics_df["alpha"] != "")
        ]

        if len(subset):
            best_idx = subset["rmse"].idxmin()

            metrics_df.loc[
                best_idx,
                "selected_alpha",
            ] = True

    metrics_name = "regression_tt_metrics.csv"

    if os.path.exists(os.path.join(RESULTS, metrics_name)):
        existing = pd.read_csv(
            os.path.join(RESULTS, metrics_name)
        )

        existing = existing[
            ~(
                (existing["solver"] == args.solver)
                & (
                    existing["target"]
                    == ("log" if args.log_target else "raw")
                )
            )
        ]

        metrics_df = pd.concat(
            [existing, metrics_df],
            ignore_index=True,
        )

    metrics_df.to_csv(
        os.path.join(RESULTS, metrics_name),
        index=False,
    )

    neighborhoods_name = (
        f"regression_neighborhood_concentration"
        f"{suffix}_{args.solver}.csv"
    )

    pd.DataFrame(neighborhood_rows).to_csv(
        os.path.join(RESULTS, neighborhoods_name),
        index=False,
    )

    if diagnostics_rows:
        diag_name = "log_target_diagnostics.csv"

        diag_df = pd.DataFrame(diagnostics_rows)

        if os.path.exists(os.path.join(RESULTS, diag_name)):
            existing = pd.read_csv(
                os.path.join(RESULTS, diag_name)
            )

            existing = existing[
                existing["solver"] != args.solver
            ]

            diag_df = pd.concat(
                [existing, diag_df],
                ignore_index=True,
            )

        diag_df.to_csv(
            os.path.join(RESULTS, diag_name),
            index=False,
        )

        print("scris:", os.path.join("results", diag_name))

    print("scris:", os.path.join("results", metrics_name))
    print("scris:", os.path.join("results", neighborhoods_name))


if __name__ == "__main__":
    main()
