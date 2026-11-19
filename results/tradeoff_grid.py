import argparse
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


SEED = 42  
FIG_W, FIG_H = (
    7.16,
    5.9,
)  
NCOLS = 3  
CONFIGS = (
    None  
)
COLORBAR = "right"  
FONT = "Liberation Serif"  
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_CBAR = 14, 13, 10, 13, 13
PT_SIZE, PT_ALPHA = 10, 0.6  
STAR_SIZE = 125
ACC_RANGE = (0.55, 0.95) 
DPI = 300


ALL = [
    "Compas_sex",
    "Compas_race",
    "German_sex",
    "heart_age",
    "student_sex",
    "Bank_age",
    "Default_sex",
    "Adult_sex",
    "Adult_race",
]
NAMES = dict(
    zip(
        ALL,
        [
            ("Compas", "sex"),
            ("Compas", "race"),
            ("German", "sex"),
            ("Heart", "age"),
            ("Student", "sex"),
            ("Bank", "age"),
            ("Default", "sex"),
            ("Adult", "sex"),
            ("Adult", "race"),
        ],
    )
)
CMAP = LinearSegmentedColormap.from_list(
    "ref",
    [
        "#2177b3",
        "#3a7aa4",
        "#689cb3",
        "#a9dada",
        "#e1c34d",
        "#feb705",
        "#fb8700",
    ],
)
GREEN, STAR = "#008000", "#ffc107"


def main():
  
  master_csv = "merged_master_log.csv"
  round_csv = "merged_round_history_log.csv"
  out_prefix = "fig_tradeoff_grid"

  keep = CONFIGS or ALL
  m = pd.read_csv(master_csv)
  r = pd.read_csv(round_csv)
  m = m[(m.experiment_type == "main") & m.dataset.isin(keep)]
  r = r[(r.experiment_type == "main") & r.dataset.isin(keep)]

  plt.rcParams.update({
      "font.family": [FONT, "DejaVu Serif"],
      "mathtext.fontset": "stix",
      "pdf.fonttype": 42,
      "axes.linewidth": 0.6,
      "axes.edgecolor": "#333",
      "xtick.major.width": 0.5,
      "ytick.major.width": 0.5,
      "xtick.major.size": 2.5,
      "ytick.major.size": 2.5,
  })
  di = lambda v: (1 - v).abs()
  norm = Normalize(*ACC_RANGE)

  nrows = int(np.ceil(len(keep) / NCOLS))
  fig, axs = plt.subplots(
    nrows, NCOLS,
    figsize=(FIG_W, FIG_H),
    squeeze=False
)
  fig.subplots_adjust(
    left=0.09,
    right=0.97,
    top=0.90,
    bottom=0.10,
    wspace=0.35,
    hspace=0.45
)
  axs = axs.ravel()

  def candidates(h):  
    p = pd.concat([
        h[["global_acc", "global_bacc", "global_di1"]].set_axis(
            ["acc", "bacc", "di1"], axis=1
        ),
        h[["best_client_acc", "best_client_bacc", "best_client_di1"]].set_axis(
            ["acc", "bacc", "di1"], axis=1
        ),
    ])
    p["DI"] = di(p.di1)
    return p

  for i, (ax, k) in enumerate(zip(axs, keep)):
    h = r[r.dataset == k]
    pts, p1 = candidates(h), candidates(h[h.seed == SEED])
    o = np.argsort(
        -pts.acc.values
    )  
    ax.scatter(
        pts.bacc.values[o],
        pts.DI.values[o],
        c=pts.acc.values[o],
        cmap=CMAP,
        norm=norm,
        s=PT_SIZE,
        alpha=PT_ALPHA,
        edgecolor="none",
        zorder=2,
    )

    s = m[(m.dataset == k) & (m.seed == SEED)].iloc[
        0
    ]  
    sb, sd = s.bacc, abs(1 - s.di1)

    P = (
        p1[["bacc", "DI"]]
        .drop_duplicates()
        .sort_values(["bacc", "DI"], ascending=[False, True])
        .values
    )
    fr, best = [], np.inf
    for b, d in P: 
      if d < best:
        fr.append((b, d))
        best = d
    if not any(np.isclose(b, sb) and np.isclose(d, sd) for b, d in fr):
      fr.append((sb, sd))  
    fr = np.array(sorted(fr))
    ax.plot(
        fr[:, 0],
        fr[:, 1],
        color=GREEN,
        lw=1.6,
        marker="o",
        ms=3,
        zorder=4,
    )
    ax.scatter(
        sb,
        sd,
        s=STAR_SIZE,
        marker="*",
        color=STAR,
        edgecolor="k",
        lw=0.7,
        zorder=6,
    )

    top = pts.DI.max() * 1.06
    ax.set_ylim(-0.025 * top, top)
    lo, up = pts.bacc.min(), pts.bacc.max()
    pad = (up - lo) * 0.05
    ax.set_xlim(lo - pad, up + pad)


    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.grid(color="#bbb", lw=0.4, ls=":")
    ax.set_axisbelow(True)

    ax.set_title(
    "%s (%s)" % NAMES[k],
    fontsize=FS_TITLE,
    fontweight="normal",
    pad=3
)

    ax.tick_params(labelsize=FS_TICK, pad=1.5)

    for sp in ("top", "right"):
      ax.spines[sp].set_visible(False)



    if i == len(keep) - 2:
      ax.set_xlabel(
          r"Balanced accuracy",
          fontsize=FS_LABEL,
          labelpad=3,
          fontweight="bold"
      )
  fig.supylabel(
    r"D I",
    fontsize=FS_LABEL,
    fontweight="bold",
    x=0.025,
    y=0.50,
    rotation=90,
    va="center"
)
  for ax in axs[len(keep) :]:
    ax.set_visible(False)

  cb = fig.colorbar(
      plt.cm.ScalarMappable(norm=norm, cmap=CMAP),
      ax=list(axs[: len(keep)]),
      location=COLORBAR,
      shrink=0.9 if COLORBAR == "right" else 0.6,
      pad=0.02,
      aspect=30 if COLORBAR == "right" else 40,
  )
  cb.set_label("Accuracy", fontsize=FS_CBAR, fontweight="bold")
  cb.ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.5)
  cb.outline.set_linewidth(0.4)

  fig.legend(
      handles=[
          Line2D(
              [0],
              [0],
              color=GREEN,
              lw=1.1,
              marker="o",
              ms=2.6,
              label="Pareto frontier",
          ),
          Line2D(
              [0],
              [0],
              marker="*",
              color="none",
              markerfacecolor=STAR,
              markeredgecolor="k",
              markersize=12,
              label="Selected model",
          ),
      ],
      loc="upper center",
      bbox_to_anchor=(0.5, 1.05),
      ncol=2,
      frameon=False,
      fontsize=FS_LEGEND,
      handletextpad=0.5,
      columnspacing=2.0,
  )

  for ext in ("png", "pdf"):
    fig.savefig(f"{out_prefix}.{ext}", dpi=DPI, bbox_inches="tight")
  print("Saved successfully as", out_prefix + ".png/.pdf")


if __name__ == "__main__":
  main()
