import streamlit as st

from core.config import show_fig, graph_caption
from modules.ehi import compute_ehi, classify_ehi, plot_ehi_trend


# ═════════════════════════════════════════════════════════
#  TAB 3 — ECONOMIC HEALTH INDEX
# ═════════════════════════════════════════════════════════
def render_tab3(df):
    st.markdown(
        '<div class="section-header">💹 Output 3 — Economic Health Index (EHI)</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this output does — Economic Health Index:</b><br>
    This module creates a <b>custom composite Economic Health Index (EHI)</b> — a single numeric score
    per year that summarises the overall economic health of India using three key indicators.<br><br>
    <b>Formula:</b> <code>EHI = 0.4 × GDP Growth − 0.3 × Inflation − 0.3 × Unemployment</code><br>
    &nbsp;&nbsp;• GDP Growth contributes positively (40% weight) — higher growth = healthier economy.<br>
    &nbsp;&nbsp;• Inflation and Unemployment contribute negatively (30% each) — high values drag the score down.<br><br>
    <b>Normalisation:</b> Raw EHI is rescaled to a 0–1 range using Min-Max Scaling for easy year-to-year comparison.<br><br>
    <b>Categories assigned per year:</b><br>
    &nbsp;&nbsp;• 🟢 <b>Strong</b> — EHI ≥ 0.70 &nbsp;|&nbsp; 🔵 <b>Stable</b> — 0.45–0.70
    &nbsp;|&nbsp; 🟠 <b>Risky</b> — 0.25–0.45 &nbsp;|&nbsp; 🔴 <b>Recession</b> — &lt; 0.25<br><br>
    <b>Why a custom index?</b> A single composite score is easier to track and communicate than
    monitoring 3 separate indicators individually.
    </div>
    """, unsafe_allow_html=True)

    ehi_df = compute_ehi(df)

    c1, c2, c3, c4 = st.columns(4)
    latest_ehi = ehi_df["EHI"].iloc[-1]
    avg_ehi    = ehi_df["EHI"].mean()
    best_yr    = int(ehi_df.loc[ehi_df["EHI"].idxmax(), "Time"])
    worst_yr   = int(ehi_df.loc[ehi_df["EHI"].idxmin(), "Time"])
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{latest_ehi:.3f}</div><div class="metric-label">Latest EHI Score</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_ehi:.3f}</div><div class="metric-label">50-Year Average EHI</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">{best_yr}</div><div class="metric-label">Best Economic Year</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-value">{worst_yr}</div><div class="metric-label">Worst Economic Year</div></div>', unsafe_allow_html=True)

    st.markdown("#### 📈 EHI Trend — 50 Years")
    fig = plot_ehi_trend(ehi_df)
    show_fig(fig)
    latest_condition = classify_ehi(latest_ehi)
    graph_caption(
        f"Top chart: EHI score from {int(ehi_df['Time'].min())} to {int(ehi_df['Time'].max())}. "
        f"Current (latest) EHI = <b>{latest_ehi:.3f}</b> → economy classified as <b>{latest_condition}</b>. "
        f"Best year = <b>{best_yr}</b> (highest EHI), Worst year = <b>{worst_yr}</b> (lowest EHI). "
        f"Bottom bar: each year's colour shows its economic condition category at a glance."
    )

    st.markdown("#### 📋 Year-wise EHI Score Table")
    ehi_display = ehi_df[["Time", "GDP growth (annual %)",
                           "Inflation, consumer prices (annual %)",
                           "Unemployment, total", "EHI_raw", "EHI"]].copy()
    ehi_display["Condition"] = ehi_display["EHI"].apply(classify_ehi)
    ehi_display.columns = ["Year", "GDP Growth %", "Inflation %",
                            "Unemployment %", "EHI Raw", "EHI (0-1)", "Condition"]
    st.dataframe(
        ehi_display.style.format({
            "GDP Growth %": "{:.2f}", "Inflation %": "{:.2f}",
            "Unemployment %": "{:.2f}", "EHI Raw": "{:.3f}", "EHI (0-1)": "{:.3f}",
        }).background_gradient(subset=["EHI (0-1)"], cmap="RdYlGn"),
        use_container_width=True
    )
    graph_caption(
        "This table shows how the EHI score is calculated for each year. "
        "Green = healthier economy (high EHI), Red = weaker economy (low EHI). "
        "The 'Condition' column gives the final economic label assigned to that year."
    )
