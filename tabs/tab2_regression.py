import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from core.config import C, show_fig, graph_caption
from modules.regression import (
    FEATURES, prepare_regression_data, evaluate_model,
    plot_feature_importance, plot_actual_vs_predicted
)


# ═════════════════════════════════════════════════════════
#  TAB 2 — REGRESSION MODELING
# ═════════════════════════════════════════════════════════
def render_tab2(df, test_size):
    st.markdown(
        '<div class="section-header">🔮 Output 2 — Multivariate Regression Modeling</div>',
        unsafe_allow_html=True
    )
    st.markdown(f"""
    <div class="desc-box">
    <b>What this output does — Regression:</b><br>
    This module trains three regression models to <b>predict GDP Growth (%)</b> using 8 other
    economic indicators as input features — simulating how the economy's GDP responds to
    changes in inflation, unemployment, trade, and other factors.<br><br>
    <b>Features used as input:</b> GDP Deflator, Inflation, Unemployment, Population Growth,
    Population Total, Real Interest Rate, Government Debt, Trade.<br>
    <b>Target variable:</b> GDP Growth (annual %).<br>
    <b>Split:</b> {int((1-test_size)*100)}% Training / {int(test_size*100)}% Testing.<br><br>
    <b>Models trained:</b><br>
    &nbsp;&nbsp;• <b>Random Forest Regressor</b> — ensemble of decision trees, handles non-linear relationships.<br>
    &nbsp;&nbsp;• <b>XGBoost Regressor</b> — gradient boosting, high accuracy on structured data.<br>
    &nbsp;&nbsp;• <b>Linear Regression</b> — simple baseline, shows direct coefficient impact.<br><br>
    <b>Metrics used:</b> R² Score (higher = better), MAE and RMSE (lower = better).
    </div>
    """, unsafe_allow_html=True)

    X, y = prepare_regression_data(df)

    if len(X) < 10:
        st.warning("Not enough data rows for regression after dropping NaNs.")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

        models_reg = {
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
        }
        if XGBOOST_AVAILABLE:
            models_reg["XGBoost"] = XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                random_state=42, eval_metric="rmse", verbosity=0)

        results_reg  = []
        model_objects = {}
        with st.spinner("Training regression models..."):
            for name, mdl in models_reg.items():
                res = evaluate_model(name, mdl, X_tr, X_te, y_tr, y_te)
                results_reg.append(res)
                model_objects[name] = res

        # ── Metrics Table ──────────────────────────────────
        st.markdown("#### 📊 Model Performance Comparison")
        metrics_df = pd.DataFrame([
            {"Model": r["Model"], "R² Score": r["R²"], "MAE": r["MAE"], "RMSE": r["RMSE"]}
            for r in results_reg
        ])
        st.dataframe(
            metrics_df.style
                .highlight_max(subset=["R² Score"], color="#c8e6c9")
                .highlight_min(subset=["MAE", "RMSE"], color="#c8e6c9"),
            use_container_width=True
        )
        best_r = max(results_reg, key=lambda r: r["R²"])
        graph_caption(
            f"Best model: <b>{best_r['Model']}</b> — R²={best_r['R²']}, MAE={best_r['MAE']}%, "
            f"RMSE={best_r['RMSE']}%. R² closer to 1.0 = model explains more variance in GDP growth. "
            f"MAE = average prediction error in percentage points. RMSE penalises large errors more."
        )

        # ── Actual vs Predicted ────────────────────────────
        st.markdown("#### 📈 Actual vs Predicted — GDP Growth (%)")
        pred_cols = st.columns(len(results_reg))
        plot_colors = [C["blue"], C["red"], C["green"]]
        for i, res in enumerate(results_reg):
            with pred_cols[i]:
                fig = plot_actual_vs_predicted(
                    res["y_test"], res["y_pred"], res["Model"],
                    plot_colors[i % len(plot_colors)]
                )
                show_fig(fig)
                graph_caption(
                    f"<b>{res['Model']}</b>: Each dot = one test year. Dots on the dashed diagonal = perfect prediction. "
                    f"R²={res['R²']} — the model explains {res['R²']*100:.1f}% of GDP growth variation."
                )

        # ── Feature Importance ─────────────────────────────
        st.markdown("#### 🔍 Feature Importance — What Drives GDP Growth?")
        fi_models = ["Random Forest"] + (["XGBoost"] if XGBOOST_AVAILABLE else [])
        fi_cols   = st.columns(len(fi_models))
        fi_colors = [C["blue"], C["red"]]
        for i, name in enumerate(fi_models):
            if name in model_objects:
                mdl      = model_objects[name]["model"]
                imp      = mdl.feature_importances_
                top_feat = list(X.columns)[np.argmax(imp)]
                with fi_cols[i]:
                    fig = plot_feature_importance(imp, list(X.columns),
                                                   f"{name} — Feature Importance", fi_colors[i])
                    show_fig(fig)
                    graph_caption(
                        f"<b>{name}</b>: Most influential feature is <b>{top_feat}</b> "
                        f"(score={max(imp):.3f}). Longer bars = stronger influence on GDP growth prediction."
                    )

        # ── Linear Regression Coefficients ────────────────
        st.markdown("#### 📋 Linear Regression — Coefficients")
        lr_mdl  = model_objects["Linear Regression"]["model"]
        coef_df = pd.DataFrame({
            "Feature": FEATURES,
            "Coefficient": lr_mdl.coef_.round(4)
        }).sort_values("Coefficient", key=abs, ascending=False)
        st.dataframe(coef_df, use_container_width=True)
        top_c = coef_df.iloc[0]
        graph_caption(
            f"Coefficients show the direct linear impact on GDP growth. "
            f"Strongest predictor: <b>{top_c['Feature']}</b> (coefficient={top_c['Coefficient']}). "
            f"Positive coefficient = variable increases GDP; Negative = it reduces GDP."
        )
