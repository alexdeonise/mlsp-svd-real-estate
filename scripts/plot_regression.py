"""Figuri pentru regresia cu evaluare train/test (M2).

Citeste:
  results/regression_tt_metrics.csv                  metrici pe train/test
  results/least_squares_top50_*_custom.csv           top residualuri pe test
  results/regression_neighborhood_concentration_custom.csv
  results/log_ls_test_predictions.csv                investigatia R² negativ

Scrie:
  figures/regression_metrics.png                     RMSE/MAE/R² pe test, raw vs log
  figures/regression_train_vs_test.png               metrici train vs test
  figures/regression_residual_distribution.png
  figures/regression_neighborhood_concentration.png
  figures/regression_log_ls_investigation.png        actual vs predictit log-LS
"""

import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
)

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

METRICS = ["rmse", "mae", "r2"]
TITLES = ["RMSE", "MAE", "R²"]


def plot_metrics():
    df = pd.read_csv(
        os.path.join(
            RESULTS,
            "regression_tt_metrics.csv"
        )
    )

    df = df[
        (df["solver"] == "custom")
        & (df["split"] == "test")
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5)
    )

    for target, color in [
        ("raw", "steelblue"),
        ("log", "darkorange"),
    ]:
        sub = df[
            df["target"] == target
        ]

        labels = sub["model"].tolist()

        for i, metric in enumerate(METRICS):
            axes[i].bar(
                labels,
                sub[metric],
                alpha=0.6,
                label=target,
                color=color,
            )

    for i, metric in enumerate(METRICS):
        axes[i].set_title(
            f"{TITLES[i]} by model (test)"
        )

        axes[i].set_ylabel(
            TITLES[i]
        )

        axes[i].tick_params(
            axis="x",
            rotation=45,
        )

    axes[0].legend()

    fig.tight_layout()

    out = os.path.join(
        FIGURES,
        "regression_metrics.png"
    )

    plt.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("scris:", out)


def plot_train_vs_test():
    df = pd.read_csv(
        os.path.join(
            RESULTS,
            "regression_tt_metrics.csv"
        )
    )

    df = df[df["solver"] == "custom"]

    models = df["model"].unique().tolist()

    x = range(len(models))

    width = 0.35

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 8)
    )

    for row, target in enumerate(["raw", "log"]):
        sub = df[df["target"] == target]

        train = sub[
            sub["split"] == "train"
        ].set_index("model")

        test = sub[
            sub["split"] == "test"
        ].set_index("model")

        for i, metric in enumerate(METRICS):
            axes[row][i].bar(
                [
                    xi - width / 2
                    for xi in x
                ],
                [
                    train.loc[m, metric]
                    for m in models
                ],
                width,
                label="train",
            )

            axes[row][i].bar(
                [
                    xi + width / 2
                    for xi in x
                ],
                [
                    test.loc[m, metric]
                    for m in models
                ],
                width,
                label="test",
            )

            axes[row][i].set_title(
                f"{TITLES[i]} — target {target}"
            )

            axes[row][i].set_xticks(x)
            axes[row][i].set_xticklabels(
                models,
                rotation=45,
            )

    axes[0][0].legend()

    fig.tight_layout()

    out = os.path.join(
        FIGURES,
        "regression_train_vs_test.png"
    )

    plt.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("scris:", out)


def plot_residual_distribution():
    df = pd.read_csv(
        os.path.join(
            RESULTS,
            "least_squares_top50_undervalued_custom.csv"
        )
    )

    df_over = pd.read_csv(
        os.path.join(
            RESULTS,
            "least_squares_top50_overvalued_custom.csv"
        )
    )

    all_residuals = pd.concat(
        [
            df[["residual"]],
            df_over[["residual"]]
        ],
        ignore_index=True
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        all_residuals["residual"],
        bins=25
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Residual (USD)")
    plt.ylabel("Count")
    plt.title(
        "Least squares residuals for top under/overvalued test properties"
    )

    plt.tight_layout()

    out = os.path.join(
        FIGURES,
        "regression_residual_distribution.png"
    )

    plt.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("scris:", out)


def plot_neighborhood_concentration():
    df = pd.read_csv(
        os.path.join(
            RESULTS,
            "regression_neighborhood_concentration_custom.csv"
        )
    )

    df = df[
        df["model"] == "least_squares"
    ]

    under = (
        df[
            df["group"] == "undervalued"
        ]
        .sort_values(
            "count",
            ascending=False
        )
        .head(10)
    )

    over = (
        df[
            df["group"] == "overvalued"
        ]
        .sort_values(
            "count",
            ascending=False
        )
        .head(10)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    axes[0].barh(
        under["neighborhood"],
        under["count"]
    )

    axes[0].invert_yaxis()
    axes[0].set_title(
        "Top neighborhoods - undervalued"
    )
    axes[0].set_xlabel(
        "Count in top 50 (test)"
    )

    axes[1].barh(
        over["neighborhood"],
        over["count"]
    )

    axes[1].invert_yaxis()
    axes[1].set_title(
        "Top neighborhoods - overvalued"
    )
    axes[1].set_xlabel(
        "Count in top 50 (test)"
    )

    fig.tight_layout()

    out = os.path.join(
        FIGURES,
        "regression_neighborhood_concentration.png"
    )

    plt.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("scris:", out)


def plot_log_ls_investigation():
    """De ce R² e negativ pe log(1+pret): predicții expoblansate."""
    df = pd.read_csv(
        os.path.join(
            RESULTS,
            "log_ls_test_predictions.csv"
        )
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5)
    )

    axes[0].scatter(
        df["actual_usd"],
        df["predicted_usd"],
        s=4,
        alpha=0.3,
    )

    lims = [
        0,
        max(
            df["actual_usd"].max(),
            df["predicted_usd"].max(),
        )
        * 1.05,
    ]

    axes[0].plot(
        lims,
        lims,
        "r--",
        linewidth=1,
    )

    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)

    axes[0].set_xlabel("Actual price (USD)")
    axes[0].set_ylabel("Predicted price (USD)")
    axes[0].set_title(
        "Log-target least squares: actual vs predicted (test)"
    )

    residual = (
        df["actual_usd"] - df["predicted_usd"]
    )

    axes[1].hist(
        residual[
            residual.abs()
            < residual.abs().quantile(0.99)
        ],
        bins=50,
    )

    axes[1].axvline(
        0,
        linestyle="--",
    )

    axes[1].set_xlabel(
        "Residual (USD), central 99%"
    )
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        "Residual distribution (tails clipped for display)"
    )

    fig.tight_layout()

    out = os.path.join(
        FIGURES,
        "regression_log_ls_investigation.png"
    )

    plt.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("scris:", out)


def main():
    os.makedirs(
        FIGURES,
        exist_ok=True
    )

    plot_metrics()
    plot_train_vs_test()
    plot_residual_distribution()
    plot_neighborhood_concentration()
    plot_log_ls_investigation()


if __name__ == "__main__":
    main()
