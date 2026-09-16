import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from modules.ehi import classify_ehi

# ═════════════════════════════════════════════════════════
#  MODULE 5 — CLASSIFICATION  (stratify-free, no ValueError)
# ═════════════════════════════════════════════════════════
def prepare_classification_data(ehi_df):
    ehi_df = ehi_df.copy()
    ehi_df["Label"] = ehi_df["EHI"].apply(classify_ehi)
    le = LabelEncoder()
    ehi_df["Label_enc"] = le.fit_transform(ehi_df["Label"])
    return ehi_df[["EHI"]], ehi_df["Label_enc"], le


def safe_split(X, y, test_size=0.2, random_state=42):
    """
    Simple split WITHOUT stratify.
    With 50 rows and 4 classes, stratified splitting causes
    'least populated class < 2' ValueError — this avoids it entirely.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def plot_confusion_matrix(cm, classes, title, color):
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = LinearSegmentedColormap.from_list("custom", ["white", color])
    im   = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
