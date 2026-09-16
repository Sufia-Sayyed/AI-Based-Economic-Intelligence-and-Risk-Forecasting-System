import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler

from core.config import C

# ═════════════════════════════════════════════════════════
#  MODULE 4 — ECONOMIC HEALTH INDEX
# ═════════════════════════════════════════════════════════
def compute_ehi(df):
    sub = df[["Time", "GDP growth (annual %)",
              "Inflation, consumer prices (annual %)",
              "Unemployment, total"]].copy().dropna()
    sub["EHI_raw"] = (
        0.4 * sub["GDP growth (annual %)"]
        - 0.3 * sub["Inflation, consumer prices (annual %)"]
        - 0.3 * sub["Unemployment, total"]
    )
    scaler = MinMaxScaler()
    sub["EHI"] = scaler.fit_transform(sub[["EHI_raw"]])
    return sub


def classify_ehi(val):
    if val >= 0.70:   return "Strong"
    elif val >= 0.45: return "Stable"
    elif val >= 0.25: return "Risky"
    else:             return "Recession"


def plot_ehi_trend(ehi_df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    ax = axes[0]
    ax.fill_between(ehi_df["Time"], 0.70, 1.0,  color=C["green"],  alpha=0.12, label="Strong (≥0.70)")
    ax.fill_between(ehi_df["Time"], 0.45, 0.70, color=C["blue"],   alpha=0.12, label="Stable (0.45–0.70)")
    ax.fill_between(ehi_df["Time"], 0.25, 0.45, color=C["orange"], alpha=0.12, label="Risky (0.25–0.45)")
    ax.fill_between(ehi_df["Time"], 0.0,  0.25, color=C["red"],    alpha=0.12, label="Recession (<0.25)")
    ax.plot(ehi_df["Time"], ehi_df["EHI"], color=C["navy"],
            lw=2.3, marker="o", markersize=4, label="EHI Score")
    ax.set_title("Economic Health Index — Normalized Score (0–1)")
    ax.set_ylabel("EHI Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    color_map = {"Strong": C["green"], "Stable": C["blue"],
                 "Risky": C["orange"], "Recession": C["red"]}
    cats = ehi_df["EHI"].apply(classify_ehi)
    ax2.bar(ehi_df["Time"], [1] * len(ehi_df),
            color=[color_map[c] for c in cats], width=0.85, edgecolor="white")
    ax2.set_title("Economic Condition Category per Year")
    ax2.set_yticks([])
    ax2.set_xlabel("Year")
    patches = [Patch(facecolor=v, label=k) for k, v in color_map.items()]
    ax2.legend(handles=patches, loc="upper left", fontsize=8)
    ax2.grid(False)
    fig.tight_layout(pad=2)
    return fig
