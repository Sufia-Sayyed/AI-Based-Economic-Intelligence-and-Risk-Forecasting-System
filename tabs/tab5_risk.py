import streamlit as st
import matplotlib.pyplot as plt

from core.config import C, show_fig, graph_caption
from modules.risk import detect_risks


# ═════════════════════════════════════════════════════════
#  TAB 5 — RISK DETECTION
# ═════════════════════════════════════════════════════════
def render_tab5(df):
    st.markdown(
        '<div class="section-header">⚠️ Output 5 — Risk Detection System</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this output does — Risk Detection:</b><br>
    This module implements a <b>Rule-Based Economic Risk Detection System</b> that scans the most
    recent economic data and raises alerts when critical thresholds are breached.<br><br>
    <b>Rules applied (Threshold Logic):</b><br>
    &nbsp;&nbsp;🔴 <b>High Inflation Risk</b> — Inflation &gt; 7%<br>
    &nbsp;&nbsp;🟠 <b>Employment Crisis</b> — Unemployment &gt; 8%<br>
    &nbsp;&nbsp;🔴 <b>Economic Slowdown</b> — GDP Growth &lt; 0% (negative growth)<br>
    &nbsp;&nbsp;🟠 <b>Consecutive GDP Decline</b> — GDP growth fell for 3 straight years<br>
    &nbsp;&nbsp;🟡 <b>Rising Unemployment Trend</b> — Unemployment rose for 3 straight years<br><br>
    <b>Additional outputs:</b><br>
    &nbsp;&nbsp;• A 50-year historical risk timeline showing when each threshold was breached.<br>
    &nbsp;&nbsp;• A year-by-year flag table marking each indicator's risk status.<br><br>
    <b>Why rule-based?</b> Economic risk detection doesn't always need heavy ML. Simple, transparent
    threshold rules are faster, more interpretable, and trusted by economists and policymakers.
    </div>
    """, unsafe_allow_html=True)

    alerts, latest_year = detect_risks(df)

    st.markdown(f"#### 🚨 Risk Alerts — Latest Year in Dataset: {latest_year}")
    for title, detail, cls in alerts:
        st.markdown(
            f'<div class="{cls}"><b>{title}</b><br>{detail}</div>',
            unsafe_allow_html=True
        )

    # ── Historical Risk Timeline ───────────────────────────
    st.markdown("#### 📊 Historical Risk Timeline — All 50 Years")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    risk_plot_cfg = [
        ("GDP growth (annual %)",                "GDP Growth (%)",    C["blue"],  None, 0),
        ("Inflation, consumer prices (annual %)", "Inflation (%)",    C["red"],   7,    1),
        ("Unemployment, total",                  "Unemployment (%)",  C["green"], 8,    2),
    ]
    for col, label, color, threshold, ax_idx in risk_plot_cfg:
        ax   = axes[ax_idx]
        vals = df[col]; yrs = df["Time"]
        ax.plot(yrs, vals, color=color, lw=2, marker="o", markersize=3)
        ax.fill_between(yrs, vals, 0, alpha=0.1, color=color)
        if threshold:
            ax.axhline(threshold, color=C["red"], ls="--", lw=1.3,
                       label=f"Risk Threshold ({threshold}%)")
            ax.fill_between(yrs, threshold, vals.clip(lower=threshold),
                            where=vals > threshold, alpha=0.25, color=C["red"])
        ax.set_ylabel(label, fontsize=9)
        if threshold: ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Year")
    axes[0].set_title("Economic Risk Indicators — 50-Year Historical Overview")
    fig.tight_layout()
    show_fig(fig)

    latest_row  = df.iloc[-1]
    gdp_val     = latest_row["GDP growth (annual %)"]
    inf_val     = latest_row["Inflation, consumer prices (annual %)"]
    unemp_val   = latest_row["Unemployment, total"]
    graph_caption(
        f"This chart shows all 3 risk indicators across 50 years. Red shaded zones = years where the "
        f"indicator breached the risk threshold. "
        f"In {latest_year}: GDP Growth = {gdp_val:.2f}%, "
        f"Inflation = {inf_val:.2f}% ({'⚠ above' if inf_val>7 else 'within'} 7% threshold), "
        f"Unemployment = {unemp_val:.2f}% ({'⚠ above' if unemp_val>8 else 'within'} 8% threshold)."
    )

    # ── Year-wise Risk Flag Table ──────────────────────────
    st.markdown("#### 📋 Year-wise Risk Flag Table")
    risk_df = df[["Time", "GDP growth (annual %)",
                  "Inflation, consumer prices (annual %)",
                  "Unemployment, total"]].copy()
    risk_df["Inflation Risk"]    = risk_df["Inflation, consumer prices (annual %)"].apply(
        lambda x: "🔴 HIGH" if x > 7 else "🟢 OK")
    risk_df["Unemployment Risk"] = risk_df["Unemployment, total"].apply(
        lambda x: "🟠 CRISIS" if x > 8 else "🟢 OK")
    risk_df["GDP Risk"]          = risk_df["GDP growth (annual %)"].apply(
        lambda x: "🔴 SLOWDOWN" if x < 0 else "🟢 OK")
    risk_df.columns = ["Year", "GDP Growth %", "Inflation %", "Unemployment %",
                       "Inflation Risk", "Unemployment Risk", "GDP Risk"]
    st.dataframe(risk_df, use_container_width=True)

    high_inf_yrs  = risk_df[risk_df["Inflation Risk"]    == "🔴 HIGH"]["Year"].tolist()
    crisis_yrs    = risk_df[risk_df["Unemployment Risk"] == "🟠 CRISIS"]["Year"].tolist()
    slowdown_yrs  = risk_df[risk_df["GDP Risk"]          == "🔴 SLOWDOWN"]["Year"].tolist()
    graph_caption(
        f"Risk flags across 50 years: "
        f"High Inflation years (>7%): {high_inf_yrs if high_inf_yrs else 'None'}. "
        f"Unemployment Crisis years (>8%): {crisis_yrs if crisis_yrs else 'None'}. "
        f"GDP Slowdown years (<0%): {slowdown_yrs if slowdown_yrs else 'None'}."
    )
