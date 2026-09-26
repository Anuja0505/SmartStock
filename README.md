# 📦 SmartStock — Demand Forecasting & Inventory Risk Intelligence

SmartStock is an end-to-end Data Science project that forecasts retail
product demand and converts those forecasts into inventory risk and
reorder recommendations.

The project combines time-series feature engineering, XGBoost forecasting,
walk-forward validation, inventory uncertainty estimation, and an
interactive Streamlit dashboard.

## 🚀 Live Demo

[Open SmartStock Dashboard](https://smartstock-anuja.streamlit.app)

## 🎯 Project Objective

The goal of SmartStock is to answer three practical questions:

1. How much demand should we expect?
2. Which products are at inventory risk?
3. How much inventory should be reordered?

## 🔄 Project Pipeline

Historical Retail Data
        ↓
Data Cleaning & EDA
        ↓
Time-Series Feature Engineering
        ↓
Demand Forecasting
        ↓
Model Comparison
        ↓
Walk-Forward Validation
        ↓
Inventory Risk Analysis
        ↓
Safety Stock & Reorder Point
        ↓
Explainability
        ↓
Streamlit Dashboard

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit

## 📊 Demand Forecasting

Target variable:

`Units Sold`

Features:

- Lag 1 day
- Lag 7 days
- Lag 14 days
- Lag 30 days
- 7-day rolling mean
- 30-day rolling mean
- Day of week
- Month
- Week of year

Models evaluated:

- 7-Day Rolling Mean Baseline
- Random Forest
- XGBoost

### Original Holdout Results

| Model | MAE | RMSE |
|---|---:|---:|
| 7-Day Baseline | 93.22 | 115.46 |
| Random Forest | 90.78 | 109.82 |
| XGBoost | 89.10 | 108.73 |

The original evaluation used a chronological train/test split rather
than a random split.

## 🔁 Walk-Forward Validation

To check whether model performance was consistent across different
historical periods, XGBoost was evaluated using three sequential
validation windows.

| Fold | MAE | RMSE |
|---|---:|---:|
| 1 | 89.93 | 110.63 |
| 2 | 90.91 | 111.00 |
| 3 | 89.17 | 108.94 |

Average:

- MAE: ~90.00
- RMSE: ~110.19

This evaluation preserves the chronological relationship between
training and future observations.

## 📦 Inventory Intelligence

### Inventory Coverage

Inventory coverage is calculated as:

`Inventory Coverage = Inventory Level / Predicted Demand`

Risk categories:

- Coverage < 1 → High Risk
- 1 ≤ Coverage < 2 → Medium Risk
- 2 ≤ Coverage ≤ 3 → Low Risk
- Coverage > 3 → Overstock

### Safety Stock

Safety stock incorporates forecast uncertainty at the
Store × Product level.

The project uses:

- 7-day lead time assumption
- 95% service level
- Store × Product forecast-error standard deviation

### Reorder Point

`Reorder Point = Lead-Time Demand + Safety Stock`

where:

`Lead-Time Demand = Predicted Daily Demand × Lead Time`

The recommended reorder quantity is:

`Recommended Reorder = max(Reorder Point - Inventory Level, 0)`

## 🧠 Model Explainability

XGBoost feature importance is used to understand which forecasting
features contribute most to the model.

The model considers:

- Rolling demand patterns
- Historical lag features
- Weekly timing
- Monthly timing
- Week-of-year seasonality

The feature-importance analysis indicates the relative contribution
of features to the model; it does not indicate the direction of their
effect on an individual prediction.

## 📈 Dashboard Features

The Streamlit dashboard provides:

- Inventory overview metrics
- Inventory risk distribution
- Store filtering
- Product filtering
- Inventory details
- Reorder recommendations
- Actual vs predicted demand
- High-risk inventory alerts
- Model performance metrics
- XGBoost feature importance

## 📁 Project Structure

```text
SmartStock/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── retail_store_inventory.csv
│   │
│   └── processed/
│       ├── smartstock_features.csv
│       ├── inventory_summary.csv
│       ├── test_predictions.csv
│       ├── walk_forward_results.csv
│       └── feature_importance.csv
│
├── notebooks/
│   ├── 01_EDA_SmartStock.ipynb
│   └── 02_Demand_Forecasting.ipynb
│
├── models/
│
├── requirements.txt
├── README.md
└── .gitignore