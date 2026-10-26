"""Raw vs log: Ridge cu alpha ales pe validare, metrici pe test (M2).

Citeste results/regression_tt_metrics.csv (solver=custom, split=test) si
compara pentru fiecare target modelul Ridge cu alpha selectat pe validare.
Scrie figures/regression_raw_vs_log.png.
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

selected = df[
    df["selected_alpha"] == True  # noqa: E712
]

metrics = ["rmse", "mae", "r2"]
titles = ["RMSE (test)", "MAE (test)", "R² (test)"]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4)
)

for i, metric in enumerate(metrics):
    axes[i].bar(
        ["Raw price", "Log price"],
        [
            selected[
                selected["target"] == "raw"
            ][metric].iloc[0],
            selected[
                selected["target"] == "log"
            ][metric].iloc[0],
        ]
    )
    axes[i].set_title(titles[i])

alphas = {
    target: selected[
        selected["target"] == target
    ]["alpha"].iloc[0]
    for target in ("raw", "log")
}

fig.suptitle(
    "Ridge with validation-selected alpha "
    f"(raw: α={alphas['raw']:g}, "
    f"log: α={alphas['log']:g}), test split"
)

fig.tight_layout()

out = os.path.join(
    FIGURES,
    "regression_raw_vs_log.png"
)

plt.savefig(
    out,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("scris:", out)
