# AI-Based Economic Intelligence and Risk Forecasting System

An interactive, AI- and data-science-based application for exploring historical Indian economic indicators, forecasting future trends, and identifying potential economic risks. The project uses time-series analysis, machine learning, statistical measures, and visual analytics to help users examine indicators such as GDP growth, inflation, and unemployment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Application Modules](#application-modules)
- [How It Works](#how-it-works)
- [Important Notes](#important-notes)
- [Future Improvements](#future-improvements)

## Overview

Economic indicators are interconnected, and changes in GDP growth, inflation, employment, interest rates, trade, and public debt can provide useful signals about economic conditions. This project brings several analytical methods together in a Streamlit dashboard so users can explore historical data, compare indicators, generate forecasts, and inspect potential risks.

The repository contains a modular Streamlit application (`app.py`) and a second, self-contained application implementation (`economic_intelligence_app.py`). The modular application separates data loading, analytical logic, and dashboard tabs into dedicated Python modules.

## Features

- **Time-series forecasting:** Forecasts GDP growth, consumer-price inflation, and unemployment using ARIMA models.
- **Stationarity testing:** Uses the Augmented Dickey–Fuller (ADF) test to assess time-series stationarity.
- **Regression analysis:** Trains and evaluates regression models to estimate GDP growth from selected economic indicators.
- **Economic Health Index (EHI):** Calculates a normalized composite score based on GDP growth, inflation, and unemployment.
- **Economic condition classification:** Maps EHI values to categories such as Strong, Stable, Risky, and Recession.
- **Risk detection:** Applies threshold- and trend-based rules to flag selected economic risks.
- **Correlation analysis:** Visualizes relationships among available economic indicators using a correlation heatmap.
- **Interactive dashboard:** Provides a Streamlit interface with adjustable forecast horizon and regression test split.
- **Visual analytics:** Displays charts, metrics, forecast tables, model summaries, and explanatory insights.

## Project Structure

```text
AI-Based-Economic-Intelligence-and-Risk-Forecasting-System/
├── assets/
│   └── style.css
├── core/
│   ├── __init__.py
│   ├── config.py
│   └── data_loader.py
├── modules/
│   ├── __init__.py
│   ├── classification.py
│   ├── correlation.py
│   ├── ehi.py
│   ├── forecasting.py
│   ├── regression.py
│   └── risk.py
├── tabs/
│   ├── __init__.py
│   ├── tab1_forecast.py
│   ├── tab2_regression.py
│   ├── tab3_ehi.py
│   ├── tab4_classification.py
│   ├── tab5_risk.py
│   └── tab6_correlation.py
├── PMF 50yrs data.csv
├── PPT_AI_Economic_Intelligence_India.pptx
├── app.py
├── economic_intelligence_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Python cache directories such as `__pycache__/` and compiled `.pyc` files are generated automatically and are intentionally omitted from this recommended structure.

### Key files and directories

| Path | Purpose |
|---|---|
| `app.py` | Main entry point for the modular Streamlit dashboard. |
| `economic_intelligence_app.py` | Alternative self-contained implementation of the application. |
| `core/data_loader.py` | Reads, cleans, converts, sorts, and preprocesses the uploaded CSV dataset. |
| `core/config.py` | Defines chart styling and shared visualization helpers. |
| `modules/forecasting.py` | Implements ADF testing and ARIMA forecasting utilities. |
| `modules/regression.py` | Prepares regression features, evaluates models, and creates regression plots. |
| `modules/ehi.py` | Calculates the Economic Health Index and plots its trend. |
| `modules/classification.py` | Prepares EHI-based classes and creates confusion-matrix visualizations. |
| `modules/risk.py` | Detects threshold-based and trend-based economic risk signals. |
| `modules/correlation.py` | Creates the economic-indicator correlation heatmap. |
| `tabs/` | Contains the six dashboard-tab rendering functions. |
| `assets/style.css` | Contains custom styling for the modular dashboard. |
| `PMF 50yrs data.csv` | Included historical economic dataset; the modular dashboard also allows a CSV to be uploaded through the sidebar. |
| `requirements.txt` | Lists the Python dependencies. |
| `PPT_AI_Economic_Intelligence_India.pptx` | Project presentation. |

## Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine learning:** Scikit-learn, XGBoost (listed as a dependency)
- **Time-series analysis:** Statsmodels (ARIMA and ADF test)

## Dataset

The repository includes `PMF 50yrs data.csv`, a historical dataset intended for long-term analysis of Indian economic indicators. The application expects relevant columns such as:

- `Time`
- `GDP growth (annual %)`
- `Inflation, consumer prices (annual %)`
- `Unemployment, total`
- `GDP deflator`
- `Population growth (annual %)`
- `Population, total`
- `Real interest rate (%)`
- `Central government debt`
- `Trade`

Some analyses use additional columns if they are present in the dataset. The application cleans column names, converts selected columns to numeric values, treats `".."` as missing data, sorts rows by `Time` when available, and applies forward/backward filling for missing values.

**Data note:** Forecasts and risk labels are analytical outputs based on the supplied dataset and the implemented assumptions. They should not be treated as official economic forecasts or as financial or policy advice.

## Installation

### 1. Clone the repository

Replace the repository URL below if your GitHub repository URL changes.

```bash
git clone https://github.com/Sufia-Sayyed/AI-Based-Economic-Intelligence-and-Risk-Forecasting-System.git
cd AI-Based-Economic-Intelligence-and-Risk-Forecasting-System
```

### 2. Create and activate a virtual environment (recommended)

**Windows — Command Prompt**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

From the repository root, run the modular dashboard:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually `http://localhost:8501`. Open that URL in your browser.

In the dashboard sidebar, upload `PMF 50yrs data.csv` (or another compatible CSV file), then adjust the forecast horizon and regression test split as needed.

The repository also includes a self-contained implementation that can be run separately:

```bash
streamlit run economic_intelligence_app.py
```

Use one entry point at a time. The modular `app.py` is the recommended starting point because its functionality is separated into `core/`, `modules/`, and `tabs/`.

## Application Modules

### 1. Time-Series Forecasting
Uses ARIMA to forecast GDP growth, consumer-price inflation, and unemployment. The forecasting utilities include an ADF stationarity test and confidence intervals for forecasts.

### 2. Regression Modeling
Uses selected economic indicators as input features and GDP growth as the target. The dashboard reports common regression evaluation metrics, including R², MAE, and RMSE, and visualizes model results.

### 3. Economic Health Index
Combines GDP growth, inflation, and unemployment into a weighted raw score, then scales the score to a range from 0 to 1 for comparison across the observations.

### 4. Classification
Assigns economic-condition labels based on EHI thresholds and evaluates classification results using the available workflow and visualizations.

### 5. Risk Detection
Checks the latest available observation against defined thresholds and looks for selected multi-year trends, including high inflation, elevated unemployment, negative GDP growth, and consecutive changes.

### 6. Correlation and Insights
Calculates and visualizes correlations between available economic indicators to help explore possible relationships in the dataset. Correlation does not by itself establish causation.

## How It Works

1. The user uploads a compatible economic-indicator CSV file.
2. The application preprocesses the data and prepares the relevant columns.
3. The analytical modules calculate forecasts, regression metrics, EHI scores, classifications, risk signals, and correlations.
4. The dashboard presents the results through interactive tables, charts, and explanatory text.
5. The user can change selected dashboard parameters to explore alternative forecast horizons and train/test splits.

## Important Notes

- Run commands from the repository root so relative paths such as `assets/style.css` resolve correctly.
- The dataset must contain the columns required by the selected analysis modules. Missing required columns may prevent a particular module from running.
- The exact results depend on the quality, coverage, and definitions of the source data, as well as the selected model settings.
- The included presentation is a project artifact and is not required to launch the Streamlit dashboard.
- This repository includes two application entry points. If you make changes, check which implementation you intend to maintain to avoid the two versions drifting apart.

## Future Improvements

- Add automated data validation and clearer handling of missing values.
- Compare ARIMA forecasts with additional time-series models and evaluate them using time-aware validation.
- Add model explainability and more robust backtesting.
- Add automated tests and a reproducible environment configuration.
- Add source citations and dataset provenance documentation.
- Deploy the dashboard so it can be accessed online.

## Acknowledgment

Developed as an academic AI and Data Science project exploring the use of machine learning, time-series analysis, and data visualization for economic intelligence and risk forecasting.
