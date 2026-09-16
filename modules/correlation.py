import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from core.config import C

# ═════════════════════════════════════════════════════════
#  MODULE 7 — CORRELATION HEATMAP
# ═════════════════════════════════════════════════════════
def plot_correlation_heatmap(df):
    cols = [c for c in [
        "GDP growth (annual %)", "GDP deflator",
        "Inflation, consumer prices (annual %)",
        "Unemployment, total", "Population growth (annual %)",
        "Real interest rate (%)", "Central government debt", "Trade",
    ] if c in df.columns]
    corr  = df[cols].corr()
    short = [c.replace("Inflation, consumer prices (annual %)", "Inflation CPI")
              .replace("GDP growth (annual %)", "GDP Growth")
              .replace("Population growth (annual %)", "Pop Growth")
              .replace("Real interest rate (%)", "Real Rate")
              .replace("Central government debt", "Govt Debt")
              for c in cols]
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = LinearSegmentedColormap.from_list("rg", [C["red"], "white", C["blue"]])
    im   = ax.imshow(corr.values, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.85, label="Correlation")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(short, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black" if abs(corr.values[i,j]) < 0.6 else "white")
    ax.set_title("Correlation Heatmap — Economic Indicators")
    fig.tight_layout()
    return fig
