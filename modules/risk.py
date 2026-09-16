import numpy as np

# ═════════════════════════════════════════════════════════
#  MODULE 6 — RISK DETECTION
# ═════════════════════════════════════════════════════════
def detect_risks(df):
    latest = df.iloc[-1]
    year   = int(latest.get("Time", 0))
    alerts = []

    gdp   = latest.get("GDP growth (annual %)", np.nan)
    inf_  = latest.get("Inflation, consumer prices (annual %)", np.nan)
    unemp = latest.get("Unemployment, total", np.nan)

    if not np.isnan(inf_) and inf_ > 7:
        alerts.append(("🔴 HIGH INFLATION RISK",
                        f"Inflation = {inf_:.2f}% (Threshold: >7%)", "risk-high"))
    if not np.isnan(unemp) and unemp > 8:
        alerts.append(("🟠 EMPLOYMENT CRISIS",
                        f"Unemployment = {unemp:.2f}% (Threshold: >8%)", "risk-medium"))
    if not np.isnan(gdp) and gdp < 0:
        alerts.append(("🔴 ECONOMIC SLOWDOWN",
                        f"GDP Growth = {gdp:.2f}% (Negative Growth)", "risk-high"))
    if len(df) >= 3:
        if (df["GDP growth (annual %)"].iloc[-3:].diff().dropna() < 0).all():
            alerts.append(("🟠 CONSECUTIVE GDP DECLINE",
                            "GDP growth has declined for 3 consecutive years.", "risk-medium"))
    if len(df) >= 3:
        if (df["Unemployment, total"].iloc[-3:].diff().dropna() > 0).all():
            alerts.append(("🟡 RISING UNEMPLOYMENT TREND",
                            "Unemployment rising for 3 consecutive years.", "risk-medium"))
    if not alerts:
        alerts.append(("🟢 NO CRITICAL RISKS DETECTED",
                        "All indicators are within acceptable thresholds.", "risk-low"))
    return alerts, year
