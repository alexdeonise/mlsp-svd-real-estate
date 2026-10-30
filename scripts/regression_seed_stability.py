"""Stabilitatea regresiei la alegerea seed-ului splitului (M2, robustețe).

Ruleaza pipeline-ul complet (solver custom) pe mai multe seed-uri si
raporteaza metricile finale pe test pentru least squares si pentru Ridge
cu alpha selectat pe validare. R² pe test e sensibil la cum cad proprietatile
foarte scumpe in split; acest experiment cuantifica acel raspand.

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

from regression import run_pipeline  # noqa: E402

RESULTS = os.path.join(
    ROOT,
    "results",
)

FIGURES = os.path.join(
    ROOT,
    "figures",
)

DEFAULT_SEEDS = (42, 7, 13, 123)


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

            for row in metric_rows:
                rows.append({
                    "seed": seed,
                    "target": target,
                    "model": row["model"],
                    "selected_alpha": row["selected_alpha"],
                    "split": row["split"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                    "r2": row["r2"],
                })

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

    # Figura: R² pe test pentru fiecare seed.
    test = df[
        df["split"] == "test"
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=False,
    )

    for ax, target in zip(
        axes,
        ("raw", "log"),
    ):
        sub = test[
            test["target"] == target
        ]

        models = sub["model"].unique()

        x = np.arange(len(models))

        width = 0.8 / len(args.seeds)

        for i, seed in enumerate(args.seeds):
            values = []

            for model in models:
                match = sub[
                    (sub["seed"] == seed)
                    & (sub["model"] == model)
                ]

                values.append(
                    match["r2"].iloc[0]
                    if len(match)
                    else np.nan
                )

            ax.bar(
                x + i * width - 0.4 + width / 2,
                values,
                width,
                label=f"seed {seed}",
            )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            models,
            rotation=30,
        )

        ax.set_title(
            f"Target: {target} — R² pe test"
        )

        ax.set_ylabel("R²")

        ax.grid(
            alpha=0.2,
            axis="y",
        )

    axes[0].legend()

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
        test.groupby(["target", "model"])["r2"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    print("\nR² pe test — rezumat pe seed-uri:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
