import streamlit as st
import pandas as pd
import numpy as np

# ═════════════════════════════════════════════════════════
#  MODULE 1 — DATA LOADING & PREPROCESSING
# ═════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df.replace("..", np.nan, inplace=True)

    numeric_cols = [
        "GDP growth (annual %)", "GDP deflator",
        "Inflation, consumer prices (annual %)",
        "Inflation, GDP deflator (annual %)",
        "Unemployment with basic education", "Unemployment, total",
        "Unemployment, youth female", "Unemployment, youth male",
        "Unemployment, youth total", "Population growth (annual %)",
        "Population, total", "Real interest rate (%)",
        "Central government debt", "Trade",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Time" in df.columns:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df.sort_values("Time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df
