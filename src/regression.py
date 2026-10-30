"""Regresia pretului de vanzare cu evaluare corecta pe train/test.

Protocol (M2 + corecturile de metodologie):
  1. Design matrix brut (one-hot, nestandardizat) reconstruit din CSV.
  2. Split reproducibil 80/20 (seed fix, salvat in results/regression_split.csv).
  3. Standardizare fit EXCLUSIV pe train (media/deviatia de pe train,
     aplicate si pe val/test) — niciun leakage din test in preprocessing.
  4. SVD pe matricea de train (solver interschimbabil, custom implicit).
  5. Selectia lui alpha Ridge: fiecare alpha e fitat pe train si evaluat
     pe VALIDARE; alpha castigator = minimul RMSE pe validare — testul NU
     participa la selectie.
  6. Modelul final (least squares + ridge cu alpha selectat) e reantrenat
     pe train+validation si evaluat O SINGURA DATA pe test. Metricile
     in-sample raportate sunt pentru modelul final (fit_split=trainval).
  7. Pentru --log-target predictiile sunt convertite inapoi in USD (expm1)
     INAINTE de calculul metricelor in dolari.

Scrie:
  results/regression_tt_metrics.csv            metrici (val + final)
  results/regression_split.csv                 indicii splitului + seed
  results/log_target_diagnostics.csv           diagnostic pentru R² negativ
  results/{model}_top50_{under/overvalued}{_log}_{solver}.csv  (modele finale)
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


def select_alpha(val_rows):
    """Alege alpha exclusiv pe metricile de validare.

    val_rows: lista de (model_name, alpha, metrics_dict) pentru modelele
    Ridge fitate pe train. Intoarce tupla castigatoare.
    """
    if not val_rows:
        raise ValueError("lista de candidati Ridge e goala")

    return min(
        val_rows,
        key=lambda row: row[2]["rmse"],
    )


def make_residual_table(df, y, predictions, split_idx, split_name, model_name):
    y_split = y[split_idx]

    residuals = y_split - predictions

    result = pd.DataFrame({
        "row_id": split_idx,
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

    ss_res = float(np.sum(residuals ** 2))

    order = np.argsort(-np.abs(residuals))

    top5_share = float(
        np.sum(residuals[order[:5]] ** 2) / ss_res
    ) if ss_res > 0 else 0.0

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
        "ss_res": ss_res,
        "ss_tot": float(np.sum((y_test - np.mean(y_test)) ** 2)),
    }


def run_pipeline(
    csv_path,
    solver="custom",
    log_target=False,
    seed=SEED,
    alphas=(100.0, 1000.0, 10000.0, 100000.0),
    rtol=1e-6,
    atol=0.0,
    primary=False,
):
    """Ruleaza intregul protocol si intoarce toate tabelele, fara sa scrie.

    `primary=True` marcheaza rularea principala (seed implicit): doar ea
    scrie results/log_ls_test_predictions.csv. Rularile auxiliare (alte
    seed-uri pentru stabilitate) nu suprascriu artefactele principale.

    Intoarce un dict cu: metric_rows, neighborhood_rows (cereri),
    diagnostics_rows, residual_tables, selected.
    """
    X_raw, y, df = load_design(csv_path)

    print("X:", X_raw.shape)
    print("solver:", solver)
    print("log target:", log_target)
    print("seed:", seed)

    train_idx, val_idx, test_idx = make_split(
        len(df),
        seed=seed,
    )

    trainval_idx = np.sort(
        np.concatenate([train_idx, val_idx])
    )

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
    A_trainval = X[trainval_idx]
    A_test = X[test_idx]

    if log_target:
        y_fit_all = np.log1p(y)
        suffix = "_log"
    else:
        y_fit_all = y.copy()
        suffix = ""

    target_name = "log" if log_target else "raw"

    metric_rows = []
    neighborhood_rows = []
    diagnostics_rows = []
    residual_tables = {}

    def fit_on(matrix):
        U_full, s_full, Vt_full, info = compute_svd(
            matrix,
            solver=solver,
            rtol=rtol,
            atol=atol,
        )

        print(format_svd_info(info))

        rank = info["rank"]

        return (
            U_full[:, :rank],
            s_full[:rank],
            Vt_full[:rank, :],
            info,
        )

    y_mean_train = np.mean(y_fit_all[train_idx])
    y_centered_train = y_fit_all[train_idx] - y_mean_train

    def to_usd(predictions_fit):
        if log_target:
            return np.expm1(predictions_fit)
        return predictions_fit

    # ------------------------------------------------------------------
    # Faza 1: fit pe train, selectia alpha EXCLUSIV pe validare.
    # ------------------------------------------------------------------
    print("--- faza 1: fit pe train, selectie alpha pe validare ---")

    U_tr, s_tr, Vt_tr, info_train = fit_on(A_train)

    print("numerical rank (train):", info_train["rank"])

    val_candidates = []

    model_specs = [("least_squares", None)]
    model_specs.extend(
        (f"ridge_alpha_{alpha:g}", alpha)
        for alpha in alphas
    )

    for model_name, alpha in model_specs:
        coefficients = fit_coefficients(
            U_tr,
            s_tr,
            Vt_tr,
            y_centered_train,
            alpha,
        )

        predictions_val = to_usd(
            predict_split(
                A_val,
                coefficients,
                y_mean_train,
            )
        )

        predictions_train = to_usd(
            predict_split(
                A_train,
                coefficients,
                y_mean_train,
            )
        )

        m_val = compute_metrics(
            y[val_idx],
            predictions_val,
        )

        m_train = compute_metrics(
            y[train_idx],
            predictions_train,
        )

        for split_name, m in [
            ("val", m_val),
            ("train", m_train),
        ]:
            metric_rows.append({
                "solver": solver,
                "target": target_name,
                "model": model_name,
                "alpha": "" if alpha is None else alpha,
                "fit_split": "train",
                "split": split_name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "rank": info_train["rank"],
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "seed": seed,
                "selected_alpha": False,
            })

        if alpha is not None:
            val_candidates.append(
                (model_name, alpha, m_val)
            )

        print(
            model_name + ":",
            f"val RMSE={m_val['rmse']:.3f}",
            f"R2={m_val['r2']:.6f}",
        )

    selected_name, selected_alpha, _ = select_alpha(
        val_candidates
    )

    print(
        "alpha selectat pe validare:",
        selected_alpha,
        f"({selected_name})",
    )

    # ------------------------------------------------------------------
    # Faza 2: refit pe train+validation, evaluare o singura data pe test.
    # ------------------------------------------------------------------
    print("--- faza 2: refit pe train+val, evaluare finala pe test ---")

    U_tv, s_tv, Vt_tv, info_trainval = fit_on(A_trainval)

    print(
        "numerical rank (trainval):",
        info_trainval["rank"],
    )

    y_mean_trainval = np.mean(y_fit_all[trainval_idx])
    y_centered_trainval = (
        y_fit_all[trainval_idx] - y_mean_trainval
    )

    final_specs = [
        ("least_squares", None),
        (selected_name, selected_alpha),
    ]

    for model_name, alpha in final_specs:
        coefficients = fit_coefficients(
            U_tv,
            s_tv,
            Vt_tv,
            y_centered_trainval,
            alpha,
        )

        predictions_test = to_usd(
            predict_split(
                A_test,
                coefficients,
                y_mean_trainval,
            )
        )

        predictions_trainval = to_usd(
            predict_split(
                A_trainval,
                coefficients,
                y_mean_trainval,
            )
        )

        m_test = compute_metrics(
            y[test_idx],
            predictions_test,
        )

        m_trainval = compute_metrics(
            y[trainval_idx],
            predictions_trainval,
        )

        is_selected = alpha is not None

        for split_name, m in [
            ("test", m_test),
            ("trainval", m_trainval),
        ]:
            metric_rows.append({
                "solver": solver,
                "target": target_name,
                "model": model_name,
                "alpha": "" if alpha is None else alpha,
                "fit_split": "trainval",
                "split": split_name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "rank": info_trainval["rank"],
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "seed": seed,
                "selected_alpha": (
                    is_selected and split_name == "test"
                ),
            })

        print(
            model_name + " (final):",
            f"test RMSE={m_test['rmse']:.3f}",
            f"MAE={m_test['mae']:.3f}",
            f"R2={m_test['r2']:.6f}",
        )

        residual_table = make_residual_table(
            df,
            y,
            predictions_test,
            test_idx,
            "test",
            model_name,
        )

        residual_tables[model_name] = residual_table

        if log_target:
            diagnostics_rows.append(
                log_diagnostics(
                    model_name,
                    solver,
                    y[test_idx],
                    predictions_test,
                )
            )

            if (
                model_name == "least_squares"
                and solver == "custom"
                and primary
            ):
                pd.DataFrame({
                    "row_id": test_idx,
                    "actual_usd": y[test_idx],
                    "predicted_usd": predictions_test,
                }).to_csv(
                    os.path.join(
                        RESULTS,
                        "log_ls_test_predictions.csv",
                    ),
                    index=False,
                )

        neighborhood_rows.append({
            "_residual_table": residual_table,
            "_model": model_name,
        })

    selected = {
        "selected_model": selected_name,
        "selected_alpha": selected_alpha,
        "info_train": info_train,
        "info_trainval": info_trainval,
        "split": (train_idx, val_idx, test_idx),
        "target": target_name,
        "suffix": suffix,
    }

    return {
        "metric_rows": metric_rows,
        "neighborhood_requests": neighborhood_rows,
        "diagnostics_rows": diagnostics_rows,
        "residual_tables": residual_tables,
        "selected": selected,
    }


def write_outputs(result, solver):
    """Scrie toate artefactele din rezultatul run_pipeline."""
    target_name = result["selected"]["target"]
    suffix = result["selected"]["suffix"]

    metrics_df = pd.DataFrame(result["metric_rows"])

    metrics_name = "regression_tt_metrics.csv"

    metrics_path = os.path.join(RESULTS, metrics_name)

    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)

        existing = existing[
            ~(
                (existing["solver"] == solver)
                & (existing["target"] == target_name)
            )
        ]

        metrics_df = pd.concat(
            [existing, metrics_df],
            ignore_index=True,
        )

    metrics_df.to_csv(metrics_path, index=False)

    print("scris:", os.path.join("results", metrics_name))

    neighborhood_rows = []

    for request in result["neighborhood_requests"]:
        residual_table = request["_residual_table"]
        model_name = request["_model"]

        undervalued, overvalued = save_top_residuals(
            residual_table,
            model_name,
            suffix,
            solver,
        )

        neighborhood_rows.extend(
            neighborhood_summary(
                undervalued,
                overvalued,
                model_name,
                solver,
            )
        )

    neighborhoods_name = (
        f"regression_neighborhood_concentration"
        f"{suffix}_{solver}.csv"
    )

    pd.DataFrame(neighborhood_rows).to_csv(
        os.path.join(RESULTS, neighborhoods_name),
        index=False,
    )

    print("scris:", os.path.join("results", neighborhoods_name))

    if result["diagnostics_rows"]:
        diag_name = "log_target_diagnostics.csv"

        diag_df = pd.DataFrame(result["diagnostics_rows"])

        diag_path = os.path.join(RESULTS, diag_name)

        if os.path.exists(diag_path):
            existing = pd.read_csv(diag_path)

            existing = existing[
                existing["solver"] != solver
            ]

            diag_df = pd.concat(
                [existing, diag_df],
                ignore_index=True,
            )

        diag_df.to_csv(diag_path, index=False)

        print("scris:", os.path.join("results", diag_name))


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

    result = run_pipeline(
        args.csv,
        solver=args.solver,
        log_target=args.log_target,
        seed=args.seed,
        alphas=args.alphas,
        rtol=args.rtol,
        atol=args.atol,
        primary=(args.seed == SEED),
    )

    train_idx, val_idx, test_idx = result["selected"]["split"]

    save_split(train_idx, val_idx, test_idx, args.seed)

    write_outputs(result, args.solver)


if __name__ == "__main__":
    main()
