import matplotlib.pyplot as plt
import streamlit as st

# ═════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═════════════════════════════════════════════════════════
C = {
    "blue":   "#1565c0",
    "red":    "#c62828",
    "green":  "#2e7d32",
    "orange": "#e65100",
    "purple": "#6a1b9a",
    "navy":   "#1a237e",
    "gray":   "#546e7a",
    "gold":   "#f9a825",
    "bg":     "#f5f7ff",
}

plt.rcParams.update({
    "figure.facecolor": C["bg"],
    "axes.facecolor":   C["bg"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "font.family":      "DejaVu Sans",
})


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def graph_caption(text):
    st.markdown(
        f'<div class="graph-caption">📌 <b>Insight:</b> {text}</div>',
        unsafe_allow_html=True
    )
