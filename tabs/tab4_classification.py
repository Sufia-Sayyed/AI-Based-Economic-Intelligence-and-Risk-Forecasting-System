import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from core.config import C, show_fig, graph_caption
from modules.ehi import compute_ehi, classify_ehi
from modules.classification import prepare_classification_data, safe_split, plot_confusion_matrix


# ═════════════════════════════════════════════════════════
#  TAB 4 — CLASSIFICATION
# ═════════════════════════════════════════════════════════
def render_tab4(df):
    st.markdown(
        '<div class="section-header">🏷️ Output 4 — Economic Condition Classification</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="desc-box">
    <b>What this output does — Classification:</b><br>
    This module trains three <b>machine learning classifiers</b> to automatically label each year's
    economic condition as <b>Strong, Stable, Risky, or Recession</b>, based on the EHI score
    computed in Output 3.<br><br>
    <b>Input feature:</b> EHI Score (0–1). &nbsp;&nbsp;<b>Labels:</b> Strong | Stable | Risky | Recession.<br><br>
    <b>Models trained:</b><br>
    &nbsp;&nbsp;• <b>Decision Tree</b> — creates if-else rules based on EHI score thresholds.<br>
    &nbsp;&nbsp;• <b>Random Forest Classifier</b> — ensemble of decision trees for higher accuracy.<br>
    &nbsp;&nbsp;• <b>Logistic Regression</b> — probabilistic classification with linear decision boundaries.<br><br>
    <b>Evaluation:</b> Accuracy score, confusion matrix, and full classification report (precision, recall, F1).<br><br>
    <b>Note on small dataset:</b> With only 50 data points and 4 classes, stratified splitting is skipped
    (it causes errors when some classes have only 1 sample in test set). A random 80/20 split is used instead.
    </div>
    """, unsafe_allow_html=True)

    try:
        ehi_df_cls        = compute_ehi(df)
        X_cls, y_cls, le  = prepare_classification_data(ehi_df_cls)

        if len(X_cls) < 10:
            st.warning("Not enough data for classification.")
        else:
            # No stratify — avoids 'least populated class' error on small datasets
            X_tr_c, X_te_c, y_tr_c, y_te_c = safe_split(
                X_cls, y_cls, test_size=0.2, random_state=42)

            cls_models = {
                "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            }

            # Labels that actually exist in the full dataset
            all_label_indices = list(range(len(le.classes_)))

            cls_results = {}
            with st.spinner("Training classifiers..."):
                for name, mdl in cls_models.items():
                    mdl.fit(X_tr_c, y_tr_c)
                    y_pred_c = mdl.predict(X_te_c)
                    cls_results[name] = {
                        "model":  mdl,
                        "acc":    accuracy_score(y_te_c, y_pred_c),
                        # Force confusion matrix to always be 4x4 using all label indices
                        "cm":     confusion_matrix(
                                      y_te_c, y_pred_c,
                                      labels=all_label_indices),
                        "y_pred": y_pred_c,
                        "y_test": y_te_c,
                    }

            # ── Accuracy Cards ─────────────────────────────
            st.markdown("#### 🎯 Classifier Accuracy Scores")
            acc_cols = st.columns(3)
            for i, (name, res) in enumerate(cls_results.items()):
                with acc_cols[i]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{res["acc"]*100:.1f}%</div>'
                        f'<div class="metric-label">{name}</div></div>',
                        unsafe_allow_html=True
                    )

            best_cls_name = max(cls_results, key=lambda k: cls_results[k]["acc"])
            graph_caption(
                f"Best classifier: <b>{best_cls_name}</b> — accuracy = "
                f"<b>{cls_results[best_cls_name]['acc']*100:.1f}%</b>. "
                f"It correctly predicted the economic condition (Strong/Stable/Risky/Recession) "
                f"for {cls_results[best_cls_name]['acc']*100:.1f}% of the test years."
            )

            # ── Confusion Matrices ─────────────────────────
            st.markdown("#### 🔢 Confusion Matrices")
            cm_colors   = [C["blue"], C["green"], C["purple"]]
            cm_cols     = st.columns(3)
            classes_lbl = le.classes_   # always all 4 class names
            for i, (name, res) in enumerate(cls_results.items()):
                with cm_cols[i]:
                    st.markdown(f"**{name}**")
                    fig = plot_confusion_matrix(
                        res["cm"], classes_lbl, name, cm_colors[i])
                    show_fig(fig)

            graph_caption(
                "Each confusion matrix shows correct predictions (diagonal) vs "
                "misclassifications (off-diagonal). Rows = actual condition, "
                "Columns = predicted condition. Darker diagonal = better accuracy."
            )

            # ── Classification Reports ─────────────────────
            st.markdown("#### 📋 Detailed Classification Reports")
            for name, res in cls_results.items():
                with st.expander(f"📄 {name} — Full Report", expanded=False):
                    # Only report on labels present in test+pred to avoid size mismatch
                    present_labels = sorted(
                        set(res["y_test"].tolist()) | set(res["y_pred"].tolist()))
                    present_names  = [
                        le.classes_[i] for i in present_labels
                        if i < len(le.classes_)
                    ]
                    try:
                        report = classification_report(
                            res["y_test"], res["y_pred"],
                            labels=present_labels,
                            target_names=present_names,
                            zero_division=0
                        )
                    except Exception as rep_err:
                        report = (
                            f"Report could not be generated automatically.\n"
                            f"Reason: {rep_err}\n\n"
                            f"Classes seen in test set: "
                            f"{[le.classes_[i] for i in sorted(set(res['y_test'].tolist()))]}"
                        )
                    st.code(report)

            # ── Predicted labels for all years ────────────
            st.markdown("#### 🗂️ Predicted Economic Condition — All 50 Years")
            best_mdl   = cls_results[best_cls_name]["model"]
            all_pred   = best_mdl.predict(X_cls)
            all_labels = le.inverse_transform(all_pred)
            pred_df = pd.DataFrame({
                "Year":             ehi_df_cls["Time"].values,
                "EHI Score":        ehi_df_cls["EHI"].values.round(3),
                "Actual Condition": ehi_df_cls["EHI"].apply(classify_ehi).values,
                f"Predicted ({best_cls_name})": all_labels,
            })
            st.dataframe(pred_df, use_container_width=True)
            match_count = (
                pred_df["Actual Condition"] == pred_df[f"Predicted ({best_cls_name})"]
            ).sum()
            graph_caption(
                f"Using the best model (<b>{best_cls_name}</b>), {match_count} out of "
                f"{len(pred_df)} years were correctly classified. "
                f"Compare 'Actual Condition' (EHI threshold rule) with 'Predicted' (ML model) — "
                f"mismatches show years where the model's decision boundary slightly differed."
            )

    except Exception as tab4_err:
        st.error(f"Classification module encountered an error: {tab4_err}")
        st.info("All other output tabs (Risk Detection, Correlation) are unaffected.")
