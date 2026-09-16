import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from core.config import C

# ═════════════════════════════════════════════════════════
#  MODULE 2 — ARIMA FORECASTING
# ═════════════════════════════════════════════════════════
def adf_test(series, name):
    result = adfuller(series.dropna())
    return {
        "Variable":   name,
        "ADF Stat":   round(result[0], 4),
        "p-value":    round(result[1], 4),
        "Stationary": "Yes ✅" if result[1] < 0.05 else "No ❌"
    }


def fit_arima(series):
    for order in [(1, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 0)]:
        try:
            return ARIMA(series.dropna(), order=order).fit()
        except Exception:
            continue
    return None


def forecast_arima(fit, steps):
    fc = fit.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int()


def plot_arima_forecast(series, fit, steps, title, color, ylabel, years):
    forecast_mean, conf_int = forecast_arima(fit, steps)
    last_yr      = int(years.iloc[-1])
    future_years = list(range(last_yr + 1, last_yr + steps + 1))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(years, series, color=color, lw=2.2, label="Historical", zorder=3)
    ax.plot(future_years, forecast_mean.values,
            color=C["gold"], lw=2.2, ls="--", marker="o",
            markersize=5, label="Forecast", zorder=4)
    ax.fill_between(future_years,
                    conf_int.iloc[:, 0].values,
                    conf_int.iloc[:, 1].values,
                    color=C["gold"], alpha=0.25, label="95% CI")
    ax.axvline(x=last_yr, color=C["gray"], ls=":", lw=1.2, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, forecast_mean, future_years
