"""
============================================================
  ECONOMIC INTELLIGENCE PREDICTION AND RISK ANALYSIS SYSTEM
  India — 50 Years Economic Data
  Built with Streamlit | VS Code Compatible
============================================================
"""

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from core.data_loader import load_and_preprocess

from tabs.tab1_forecast import render_tab1
from tabs.tab2_regression import render_tab2
from tabs.tab3_ehi import render_tab3
from tabs.tab4_classification import render_tab4
from tabs.tab5_risk import render_tab5
from tabs.tab6_correlation import render_tab6

# ═════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Economic Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════
#  CUSTOM CSS  (loaded from assets/style.css — same rules as before)
# ═════════════════════════════════════════════════════════
with open("assets/style.css") as _css_file:
    st.markdown(f"<style>\n{_css_file.read()}\n</style>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Flag_of_India.svg", width=80)
    st.markdown("## 🧠 Economic Intelligence System")
    st.markdown("India — 50 Years Economic Data")
    st.divider()

    uploaded_file  = st.file_uploader("📂 Upload CSV Dataset", type=["csv"],
                                       help="Upload PMF 50yrs data.csv")
    forecast_steps = st.slider("Forecast Horizon (Years)", 3, 10, 5)
    test_size      = st.slider("Regression Test Split (%)", 10, 40, 20) / 100

    st.divider()
    st.markdown("""
    **Modules:**
    1. 📈 Time Series Forecast
    2. 🔮 Regression Modeling
    3. 💹 Economic Health Index
    4. 🏷️ Classification System
    5. ⚠️ Risk Detection
    6. 🔗 Correlation & Insights
    """)
    st.divider()
    st.caption("© 2025 Economic Intelligence System")


# ═════════════════════════════════════════════════════════
#  MAIN HEADER
# ═════════════════════════════════════════════════════════
st.markdown(
    '<div class="main-title">📊 Economic Intelligence Prediction & Risk Analysis System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">India — 50 Years of Economic Data | AI & Data Science Project</div>',
    unsafe_allow_html=True
)

if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
    👈 <b>Upload your CSV dataset</b> from the sidebar to begin.<br>
    Expected file: <code>PMF 50yrs data.csv</code>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Analysis Modules</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><div class="metric-value">50</div><div class="metric-label">Years of Data</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><div class="metric-value">18</div><div class="metric-label">Economic Indicators</div></div>', unsafe_allow_html=True)
    st.stop()


# ── Load Data ─────────────────────────────────────────────
with st.spinner("Loading and preprocessing dataset..."):
    df = load_and_preprocess(uploaded_file)

st.success(f"✅ Dataset loaded: {len(df)} rows × {len(df.columns)} columns")

# ── KPI Cards ─────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
yr_range = f"{int(df['Time'].min())}–{int(df['Time'].max())}" if "Time" in df.columns else "N/A"
with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{yr_range}</div><div class="metric-label">Year Range</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">{df["GDP growth (annual %)"].mean():.2f}%</div><div class="metric-label">Avg GDP Growth</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><div class="metric-value">{df["Inflation, consumer prices (annual %)"].mean():.2f}%</div><div class="metric-label">Avg Inflation</div></div>', unsafe_allow_html=True)
with c5: st.markdown(f'<div class="metric-card"><div class="metric-value">{df["Unemployment, total"].mean():.2f}%</div><div class="metric-label">Avg Unemployment</div></div>', unsafe_allow_html=True)

with st.expander("🔍 View Raw Dataset", expanded=False):
    st.dataframe(df, use_container_width=True)


# ═════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 OUTPUT 1 — Forecast",
    "🔮 OUTPUT 2 — Regression",
    "💹 OUTPUT 3 — EHI",
    "🏷️ OUTPUT 4 — Classification",
    "⚠️ OUTPUT 5 — Risk Detection",
    "🔗 Correlation & Insights",
])

# ═════════════════════════════════════════════════════════
#  TAB 1 — TIME SERIES FORECAST
# ═════════════════════════════════════════════════════════
with tabs[0]:
    render_tab1(df, forecast_steps)

# ═════════════════════════════════════════════════════════
#  TAB 2 — REGRESSION MODELING
# ═════════════════════════════════════════════════════════
with tabs[1]:
    render_tab2(df, test_size)

# ═════════════════════════════════════════════════════════
#  TAB 3 — ECONOMIC HEALTH INDEX
# ═════════════════════════════════════════════════════════
with tabs[2]:
    render_tab3(df)

# ═════════════════════════════════════════════════════════
#  TAB 4 — CLASSIFICATION
# ═════════════════════════════════════════════════════════
with tabs[3]:
    render_tab4(df)

# ═════════════════════════════════════════════════════════
#  TAB 5 — RISK DETECTION
# ═════════════════════════════════════════════════════════
with tabs[4]:
    render_tab5(df)

# ═════════════════════════════════════════════════════════
#  TAB 6 — CORRELATION & INSIGHTS
# ═════════════════════════════════════════════════════════
with tabs[5]:
    render_tab6(df)


# ═════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style='text-align:center; color:#9e9e9e; font-size:0.88rem; padding:1rem 0;'>
    <b>Economic Intelligence Prediction & Risk Analysis System</b><br>
    India — 50-Year Economic Dataset | Built with Streamlit · Scikit-learn · Statsmodels · Matplotlib
</div>
""", unsafe_allow_html=True)
