
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MASTER_CSV = "merged_master_log.csv"
OUT_PREFIX = "fig_selection_gain"

FIG_W, FIG_H = 3.5, 3.15


CONFIGS = None  

FONT = "Liberation Serif"

FS_TICK = 8
FS_LABEL = 8.5
FS_LEGEND = 8

MARK_SEL = 38
MARK_FIN = 34

COL_SEL = "#2177b3"
COL_FIN = "#fb8700"

FLOOR = 1e-3
XLIM = (6e-4, 0.8)

MARGINS = dict(
    left=0.285,
    right=0.985,
    bottom=0.15,
    top=0.91
)

DPI = 300


ALL = [
    'Compas_sex',
    'Compas_race',
    'German_sex',
    'heart_age',
    'student_sex',
    'Bank_age',
    'Default_sex',
    'Adult_sex',
    'Adult_race'
]

LABELS = dict(zip(
    ALL,
    [
        "Compas (sex)",
        "Compas (race)",
        "German",
        "Heart",
        "Student",
        "Bank",
        "Default",
        "Adult (sex)",
        "Adult (race)"
    ]
))


def main():

    keep = CONFIGS or ALL

    # Load master log
    m = pd.read_csv(MASTER_CSV)

    # Retain ablation runs
    m = m[
        (m.experiment_type == "ablation")
        & m.dataset.isin(keep)
        & m.variant.isin(["full", "final_round_only"])
    ].copy()

    
    m["DI"] = (1 - m.di1).abs()

    
    g = (
        m.pivot_table(
            index=["dataset", "seed"],
            columns="variant",
            values="DI"
        )
        .groupby("dataset")
        .mean()
        .loc[keep]
    )

   
    plt.rcParams.update({
        "font.family": [FONT, "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    y = np.arange(len(keep))[::-1]

    
    for yi, k in zip(y, keep):

        s = max(g.loc[k, "full"], FLOOR)
        f = max(g.loc[k, "final_round_only"], FLOOR)

        
        ax.plot(
            [s, f],
            [yi, yi],
            color="#b9b9b9",
            lw=1.3,
            zorder=1
        )

        
        ax.scatter(
            f,
            yi,
            s=MARK_FIN,
            marker="s",
            color=COL_FIN,
            edgecolor="white",
            lw=0.5,
            zorder=3
        )

       
        ax.scatter(
            s,
            yi,
            s=MARK_SEL,
            marker="o",
            color=COL_SEL,
            edgecolor="white",
            lw=0.5,
            zorder=4
        )

    
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)

   
    ax.set_yticks(y)
    ax.set_yticklabels(
        [LABELS[k] for k in keep],
        fontsize=FS_TICK
    )

   
    ax.tick_params(
        axis="x",
        labelsize=FS_TICK,
        pad=1.5
    )

    
    ax.set_xlabel(
        r"Mean $|\mathrm{DI}|$ over five seeds (lower is fairer)",
        fontsize=FS_LABEL,
        labelpad=2
    )

    
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

   
    ax.grid(
        axis="x",
        color="#d0d0d0",
        lw=0.4,
        ls=":"
    )

    ax.set_axisbelow(True)

    
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COL_SEL,
                markeredgecolor="white",
                markersize=6,
                label="Selected model"
            ),

            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=COL_FIN,
                markeredgecolor="white",
                markersize=6,
                label="Final-round aggregate"
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.42, 1.0),
        ncol=2,
        frameon=False,
        fontsize=FS_LEGEND,
        handletextpad=0.2,
        columnspacing=1.0
    )

    
    fig.subplots_adjust(**MARGINS)

    
    for ext in ("png", "pdf"):

        fig.savefig(
            f"{OUT_PREFIX}.{ext}",
            dpi=DPI,
            bbox_inches="tight"
        )

    plt.close(fig)

    print("Saved successfully:")
    print(f"{OUT_PREFIX}.png")
    print(f"{OUT_PREFIX}.pdf")


main()
