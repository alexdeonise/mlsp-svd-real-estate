import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
)

FULL_PATH = os.path.join(
    ROOT,
    "results",
    "rank_k_full.csv"
)

NUMERIC_PATH = os.path.join(
    ROOT,
    "results",
    "rank_k_numeric.csv"
)

OUT_PATH = os.path.join(
    ROOT,
    "figures",
    "rank_k_comparison.png"
)


def first_k_at_threshold(df, threshold):
    return int(
        df.loc[
            df["info_kept"] >= threshold,
            "k"
        ].iloc[0]
    )


def main():
    full = pd.read_csv(FULL_PATH)
    numeric = pd.read_csv(NUMERIC_PATH)

    full_k90 = first_k_at_threshold(full, 0.90)
    full_k95 = first_k_at_threshold(full, 0.95)

    numeric_k90 = first_k_at_threshold(numeric, 0.90)
    numeric_k95 = first_k_at_threshold(numeric, 0.95)

    os.makedirs(
        os.path.join(ROOT, "figures"),
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].plot(
        full["k"],
        full["info_kept"]
    )

    axes[0].axhline(
        0.90,
        linestyle="--"
    )

    axes[0].axhline(
        0.95,
        linestyle="--"
    )

    axes[0].axvline(
        full_k90,
        linestyle=":"
    )

    axes[0].axvline(
        full_k95,
        linestyle=":"
    )

    axes[0].set_title(
        f"Full features\n90%: k={full_k90}, 95%: k={full_k95}"
    )

    axes[0].set_xlabel("Rank k")
    axes[0].set_ylabel("Information retained")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.2)

    probe_ks = (5, 10, 20, 50)

    for k in probe_ks:
        if k > len(full):
            continue

        energy = float(
            full.loc[
                full["k"] == k,
                "info_kept",
            ].iloc[0]
        )

        axes[0].plot(
            [k],
            [energy],
            marker="o",
            color="red",
        )

        axes[0].annotate(
            f"k={k}: {energy:.1%}",
            xy=(k, energy),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="red",
        )

    axes[1].plot(
        numeric["k"],
        numeric["info_kept"],
        marker="o"
    )

    axes[1].axhline(
        0.90,
        linestyle="--"
    )

    axes[1].axhline(
        0.95,
        linestyle="--"
    )

    axes[1].axvline(
        numeric_k90,
        linestyle=":"
    )

    axes[1].axvline(
        numeric_k95,
        linestyle=":"
    )

    axes[1].set_title(
        f"Numeric features only\n90%: k={numeric_k90}, 95%: k={numeric_k95}"
    )

    axes[1].set_xlabel("Rank k")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        "SVD information retained: full vs numeric features "
        "(custom solver; numpy overlap perfect la k mic)"
    )

    fig.tight_layout()

    plt.savefig(
        OUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    print("scris:", OUT_PATH)


if __name__ == "__main__":
    main()