import streamlit as st
import matplotlib.pyplot as plt

from core.config import C, show_fig, graph_caption
from modules.correlation import plot_correlation_heatmap


# ═════════════════════════════════════════════════════════
#  TAB 6 — CORRELATION & INSIGHTS
# ═════════════════════════════════════════════════════════
def render_tab6(df):
    st.markdown(
        '<div class="section-header">🔗 Correlation Analysis & Economic Insights</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this output does — Correlation & Insights:</b><br>
    This module provides a <b>deep statistical analysis</b> of how India's economic indicators
    relate to each other, and auto-generates key insights from 50 years of data.<br><br>
    <b>Outputs in this section:</b><br>
    &nbsp;&nbsp;• <b>Correlation Heatmap</b> — Pearson correlation between all indicator pairs (range: −1 to +1).
    Blue = positive correlation, Red = negative correlation.<br>
    &nbsp;&nbsp;• <b>Multi-indicator Trend Plots</b> — 50-year line charts for GDP, Inflation, Unemployment, Trade.<br>
    &nbsp;&nbsp;• <b>Automated Insights</b> — key data-driven findings such as peak years, trend direction,
    and strongest economic relationships.<br><br>
    <b>Why correlation?</b> If inflation consistently rises before GDP falls, inflation becomes a
    leading indicator — helping policymakers act before a slowdown occurs.
    </div>
    """, unsafe_allow_html=True)

    # ── Heatmap ───────────────────────────────────────────
    st.markdown("#### 🌡️ Correlation Heatmap — All Economic Indicators")
    fig = plot_correlation_heatmap(df)
    show_fig(fig)

    corr_gdp_inf   = df[["GDP growth (annual %)", "Inflation, consumer prices (annual %)"]].corr().iloc[0, 1]
    corr_gdp_unemp = df[["GDP growth (annual %)", "Unemployment, total"]].corr().iloc[0, 1]
    graph_caption(
        f"Each cell shows the Pearson correlation between two indicators. "
        f"GDP Growth vs Inflation = {corr_gdp_inf:.3f} "
        f"({'positively' if corr_gdp_inf > 0 else 'negatively'} correlated). "
        f"GDP Growth vs Unemployment = {corr_gdp_unemp:.3f} "
        f"({'positively' if corr_gdp_unemp > 0 else 'negatively'} correlated). "
        f"Values near ±1 = strong relationship; near 0 = weak/no relationship."
    )

    # ── Multi-indicator Trend Plots ────────────────────────
    st.markdown("#### 📈 Multi-Indicator Economic Trends (50 Years)")
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    trend_cfg = [
        ("GDP growth (annual %)",                "GDP Growth (%)",    C["blue"],   axes2[0, 0]),
        ("Inflation, consumer prices (annual %)", "Inflation (%)",    C["red"],    axes2[0, 1]),
        ("Unemployment, total",                  "Unemployment (%)",  C["green"],  axes2[1, 0]),
        ("Trade",                                "Trade (% of GDP)",  C["purple"], axes2[1, 1]),
    ]
    for col, label, color, ax in trend_cfg:
        if col in df.columns:
            ax.plot(df["Time"], df[col], color=color, lw=2)
            ax.fill_between(df["Time"], df[col], alpha=0.12, color=color)
            ax.set_title(label); ax.set_xlabel("Year"); ax.set_ylabel(label)
            ax.grid(True, alpha=0.2)
    fig2.suptitle("India — Key Economic Indicator Trends (50 Years)",
                  fontsize=13, fontweight="bold", y=1.01)
    fig2.tight_layout()
    show_fig(fig2)

    gdp_peak_yr  = int(df.loc[df["GDP growth (annual %)"].idxmax(), "Time"])
    gdp_peak_val = df["GDP growth (annual %)"].max()
    inf_peak_yr  = int(df.loc[df["Inflation, consumer prices (annual %)"].idxmax(), "Time"])
    inf_peak_val = df["Inflation, consumer prices (annual %)"].max()
    graph_caption(
        f"These 4 charts show how each indicator changed over 50 years. "
        f"GDP Growth peaked at <b>{gdp_peak_val:.2f}%</b> in <b>{gdp_peak_yr}</b>. "
        f"Inflation peaked at <b>{inf_peak_val:.2f}%</b> in <b>{inf_peak_yr}</b>. "
        f"Trade shows India's increasing global integration over the decades."
    )

    # ── Automated Insights ─────────────────────────────────
    st.markdown("#### 💡 Automated Economic Insights (Data-Driven)")
    insights = []

    gdp_recent = df["GDP growth (annual %)"].iloc[-5:].mean()
    if gdp_recent > 6:
        insights.append(("🟢", "Strong Recent GDP Momentum",
                          f"Average GDP growth over last 5 years: {gdp_recent:.2f}%"))
    elif gdp_recent > 3:
        insights.append(("🟡", "Moderate GDP Growth",
                          f"Average GDP growth over last 5 years: {gdp_recent:.2f}%"))
    else:
        insights.append(("🔴", "Weak GDP Growth",
                          f"Average GDP growth over last 5 years: {gdp_recent:.2f}%"))

    insights.append(("📊", "GDP–Inflation Correlation",
                      f"Pearson r = {corr_gdp_inf:.3f} "
                      f"({'positive' if corr_gdp_inf > 0 else 'negative'} relationship)"))
    insights.append(("📊", "GDP–Unemployment Correlation",
                      f"Pearson r = {corr_gdp_unemp:.3f} "
                      f"({'positive' if corr_gdp_unemp > 0 else 'negative'} relationship)"))
    insights.append(("🏆", "Best GDP Growth Year",
                      f"{gdp_peak_yr} — GDP grew at {gdp_peak_val:.2f}%"))
    insights.append(("📉", "Worst GDP Growth Year",
                      f"{int(df.loc[df['GDP growth (annual %)'].idxmin(), 'Time'])} — "
                      f"GDP was {df['GDP growth (annual %)'].min():.2f}%"))
    insights.append(("🔥", "Peak Inflation Year",
                      f"{inf_peak_yr} — Inflation reached {inf_peak_val:.2f}%"))
    insights.append(("👥", "Average Population Growth",
                      f"{df['Population growth (annual %)'].mean():.2f}% per year over 50 years"))

    for icon, title, detail in insights:
        st.markdown(f"""
        <div style="background:#f8f9ff; border:1px solid #c5cae9; border-radius:8px;
                    padding:0.8rem 1.2rem; margin:0.4rem 0;">
            <span style="font-size:1.3rem">{icon}</span>
            <b style="color:#1a237e; margin-left:0.5rem;">{title}</b>
            <br><span style="color:#546e7a; margin-left:2rem;">{detail}</span>
        </div>""", unsafe_allow_html=True)
