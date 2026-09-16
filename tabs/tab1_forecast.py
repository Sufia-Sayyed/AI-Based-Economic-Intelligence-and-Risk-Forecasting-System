import streamlit as st
import pandas as pd

from core.config import C, show_fig, graph_caption
from modules.forecasting import adf_test, fit_arima, plot_arima_forecast


# ═════════════════════════════════════════════════════════
#  TAB 1 — TIME SERIES FORECAST
# ═════════════════════════════════════════════════════════
def render_tab1(df, forecast_steps):
    st.markdown(
        '<div class="section-header">📈 Output 1 — Time Series Forecasting using ARIMA</div>',
        unsafe_allow_html=True
    )
    st.markdown(f"""
    <div class="desc-box">
    <b>What this output does — Forecast:</b><br>
    This module uses the <b>ARIMA (AutoRegressive Integrated Moving Average)</b> model to forecast
    future values of three critical economic indicators based on 50 years of historical data.<br><br>
    <b>Indicators forecasted:</b> GDP Growth (%), Inflation CPI (%), Unemployment (%).<br>
    <b>Forecast horizon:</b> Next <b>{forecast_steps} years</b>.<br><br>
    <b>Steps performed:</b>
    &nbsp;&nbsp;• ADF Stationarity Test — checks whether each series needs differencing before ARIMA is applied.<br>
    &nbsp;&nbsp;• ARIMA(1,1,1) model fitted for each indicator with automatic fallback to simpler orders.<br>
    &nbsp;&nbsp;• Forecast values + 95% confidence interval are computed and plotted.<br>
    &nbsp;&nbsp;• Historical trend and forecast are overlaid on the same chart for clear comparison.<br><br>
    <b>Why ARIMA?</b> It is the industry-standard model for yearly economic time-series.
    It handles trends, autocorrelation, and differencing — all common in macroeconomic data.
    </div>
    """, unsafe_allow_html=True)

    years = df["Time"].dropna()

    # ── ADF Tests ─────────────────────────────────────────
    st.markdown("#### 🔬 Stationarity Test (ADF) Results")
    adf_rows = []
    for col, lbl in [
        ("GDP growth (annual %)",                "GDP Growth"),
        ("Inflation, consumer prices (annual %)", "Inflation CPI"),
        ("Unemployment, total",                  "Unemployment"),
    ]:
        if col in df.columns:
            adf_rows.append(adf_test(df[col].dropna(), lbl))
    st.dataframe(pd.DataFrame(adf_rows), use_container_width=True)
    graph_caption(
        "A p-value < 0.05 means the series is stationary (no unit root) and is directly suited for ARIMA. "
        "A non-stationary series is automatically differenced (d=1) before fitting."
    )

    # ── Forecasts ─────────────────────────────────────────
    series_config = [
        ("GDP growth (annual %)",                "GDP Growth",    C["blue"],  "Growth (%)"),
        ("Inflation, consumer prices (annual %)", "Inflation CPI", C["red"],   "Inflation (%)"),
        ("Unemployment, total",                  "Unemployment",  C["green"], "Unemployment (%)"),
    ]

    forecast_table   = {}
    last_future_yrs  = []

    for col, label, color, ylabel in series_config:
        st.markdown(f"---\n#### 📊 {label} — ARIMA Forecast")
        series = df[col].dropna().reset_index(drop=True)
        yr_arr = df.loc[df[col].notna(), "Time"].reset_index(drop=True)

        with st.spinner(f"Fitting ARIMA for {label}..."):
            fit = fit_arima(series)

        if fit is None:
            st.warning(f"ARIMA could not be fitted for {label}.")
            continue

        col1, col2 = st.columns([3, 1])
        with col1:
            fig, fc_mean, future_yrs = plot_arima_forecast(
                series, fit, forecast_steps,
                f"{label} — Historical + {forecast_steps}-Year ARIMA Forecast",
                color, ylabel, yr_arr
            )
            show_fig(fig)

        with col2:
            st.markdown(f"**{label} Predicted Values**")
            fc_df = pd.DataFrame({"Year": future_yrs,
                                   "Forecast (%)": fc_mean.values.round(3)})
            st.dataframe(fc_df, use_container_width=True)
            forecast_table[label] = dict(zip(future_yrs, fc_mean.values.round(3)))

        # Caption with year-by-year prediction summary
        caption_parts = ", ".join(
            [f"{yr}: {v:.2f}%" for yr, v in zip(future_yrs, fc_mean.values)]
        )
        graph_caption(
            f"This graph forecasts <b>{label}</b> for the next {forecast_steps} years. "
            f"Predicted values → {caption_parts}. "
            f"The yellow dashed line is the ARIMA forecast; the shaded band is the 95% confidence interval "
            f"(actual value will likely fall within this range). "
            f"The vertical dotted line separates historical data from the forecast period."
        )

        with st.expander(f"📋 ARIMA Model Summary — {label}", expanded=False):
            st.text(str(fit.summary()))

        last_future_yrs = future_yrs

    if forecast_table:
        st.markdown("---\n#### 📋 Combined Forecast Summary Table")
        combined = pd.DataFrame(forecast_table, index=last_future_yrs)
        combined.index.name = "Year"
        st.dataframe(combined.style.format("{:.3f}"), use_container_width=True)
        graph_caption(
            "This combined table summarises ARIMA predictions for all three indicators together. "
            "Use this to compare whether GDP, Inflation, and Unemployment are expected to rise or fall "
            "in the same years — a useful tool for cross-indicator policy planning."
        )
