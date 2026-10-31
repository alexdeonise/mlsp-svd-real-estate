"""Stabilitatea regresiei la alegerea seed-ului splitului (M2, robustețe).

Ruleaza pipeline-ul complet (solver custom) pe mai multe seed-uri si
raporteaza metricile finale pe test pentru least squares si pentru Ridge
cu alpha selectat pe validare. R² pe test e sensibil la cum cad proprietatile
foarte scumpe in split; de aceea raportam si R² calculat pe test DIN CARE
EXCLUDEM top 1% cele mai scumpe tranzactii (r2_excl_top1pct) — daca
diferenta e mare, R² pe tot testul e dominat de acele tranzactii, nu de
calitatea modelului tipic.

Scrie:
  results/regression_seed_stability.csv
  figures/regression_seed_stability.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(
    0,
    os.path.join(
        ROOT,
        "src",
    )
)

from regression import compute_metrics, run_pipeline  # noqa: E402

RESULTS = os.path.join(
    ROOT,
    "results",
)

FIGURES = os.path.join(
    ROOT,
    "figures",
)

DEFAULT_SEEDS = (42, 7, 13, 123, 2024, 5, 99, 314)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "csv",
    )

    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )

    args = ap.parse_args()

    rows = []

    for seed in args.seeds:
        for log_target in (False, True):
            target = "log" if log_target else "raw"

            print(
                f"=== seed={seed} target={target} ==="
            )

            result = run_pipeline(
                args.csv,
                solver="custom",
                log_target=log_target,
                seed=seed,
                primary=False,
            )

            selected = result["selected"]

            metric_rows = [
                row
                for row in result["metric_rows"]
                if row["fit_split"] == "trainval"
            ]

            final_test = result["final_test"]
            y_test = final_test["y"]

            # Pragul top 1% pe pretul ACTUAL din acest test split.
            cut = np.quantile(
                y_test,
                0.99,
            )

            keep = y_test <= cut

            r2_excl = {
                model: compute_metrics(
                    y_test[keep],
                    predictions[keep],
                )["r2"]
                for model, predictions in
                final_test["predictions"].items()
            }

            for row in metric_rows:
                out_row = {
                    "seed": seed,
                    "target": target,
                    "model": row["model"],
                    "selected_alpha": row["selected_alpha"],
                    "split": row["split"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                    "r2": row["r2"],
                    "r2_excl_top1pct": (
                        r2_excl.get(row["model"])
                        if row["split"] == "test"
                        else None
                    ),
                }

                rows.append(out_row)

            print(
                "alpha selectat:",
                selected["selected_alpha"],
            )

    df = pd.DataFrame(rows)

    os.makedirs(
        RESULTS,
        exist_ok=True,
    )

    out_csv = os.path.join(
        RESULTS,
        "regression_seed_stability.csv",
    )

    df.to_csv(
        out_csv,
        index=False,
    )

    print("scris:", out_csv)

    # Figura: R² pe test pentru fiecare seed, intreg vs fara top 1% scump.
    test = df[
        df["split"] == "test"
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        sharey=False,
    )

    for row, target in enumerate(["raw", "log"]):
        sub = test[
            test["target"] == target
        ]

        models = sub["model"].unique()

        x = np.arange(len(models))

        width = 0.8 / len(args.seeds)

        for i, seed in enumerate(args.seeds):
            full_values = []
            excl_values = []

            for model in models:
                match = sub[
                    (sub["seed"] == seed)
                    & (sub["model"] == model)
                ]

                if len(match):
                    full_values.append(
                        match["r2"].iloc[0]
                    )
                    excl_values.append(
                        match["r2_excl_top1pct"].iloc[0]
                    )
                else:
                    full_values.append(np.nan)
                    excl_values.append(np.nan)

            offset = (
                x
                + i * width
                - 0.4
                + width / 2
            )

            axes[row][0].bar(
                offset,
                full_values,
                width,
                label=f"seed {seed}",
            )

            axes[row][1].bar(
                offset,
                excl_values,
                width,
                label=f"seed {seed}",
            )

        for col, title in [
            (0, "R² pe tot testul"),
            (1, "R² fără top 1% cele mai scumpe"),
        ]:
            axes[row][col].axhline(
                0,
                linestyle="--",
                linewidth=1,
            )

            axes[row][col].set_xticks(x)
            axes[row][col].set_xticklabels(
                models,
                rotation=30,
            )

            axes[row][col].set_title(
                f"Target {target} — {title}"
            )

            axes[row][col].set_ylabel("R²")

            axes[row][col].grid(
                alpha=0.2,
                axis="y",
            )

    axes[0][0].legend()

    fig.tight_layout()

    out_png = os.path.join(
        FIGURES,
        "regression_seed_stability.png",
    )

    plt.savefig(
        out_png,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("scris:", out_png)

    # Rezumat in stdout.
    summary = (
        test.groupby(["target", "model"])[
            ["r2", "r2_excl_top1pct"]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    print("\nR² pe test — rezumat pe seed-uri:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
