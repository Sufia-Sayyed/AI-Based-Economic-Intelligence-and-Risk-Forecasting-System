import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from core.config import C

# ═════════════════════════════════════════════════════════
#  MODULE 3 — REGRESSION
# ═════════════════════════════════════════════════════════
FEATURES = [
    "GDP deflator",
    "Inflation, consumer prices (annual %)",
    "Unemployment, total",
    "Population growth (annual %)",
    "Population, total",
    "Real interest rate (%)",
    "Central government debt",
    "Trade",
]
TARGET = "GDP growth (annual %)"


def prepare_regression_data(df):
    sub = df[FEATURES + [TARGET]].dropna()
    return sub[FEATURES], sub[TARGET]


def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "Model": name,
        "R²":    round(r2_score(y_te, y_pred), 4),
        "MAE":   round(mean_absolute_error(y_te, y_pred), 4),
        "RMSE":  round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
        "model": model, "y_pred": y_pred, "y_test": y_te
    }


def plot_feature_importance(importances, feature_names, title, color):
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([feature_names[i] for i in idx], importances[idx],
            color=color, edgecolor="white", height=0.65)
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.25)
    for i, v in enumerate(importances[idx]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred, model_name, color):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(y_test, y_pred, color=color, alpha=0.75,
               edgecolors="white", s=70, zorder=3)
    lims = [min(y_test.min(), y_pred.min()) - 0.5,
            max(y_test.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, color=C["gray"], ls="--", lw=1.5, label="Perfect fit")
    ax.set_title(f"Actual vs Predicted\n{model_name}")
    ax.set_xlabel("Actual GDP Growth (%)")
    ax.set_ylabel("Predicted GDP Growth (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig
