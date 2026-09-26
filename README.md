# 📦 SmartStock — Demand Forecasting & Inventory Risk Intelligence

SmartStock is an end-to-end Data Science project that uses historical
retail sales data and machine learning to forecast product demand,
identify inventory risks, and generate reorder recommendations.

## 🚀 Features

- Historical demand analysis
- Time-based feature engineering
- Demand forecasting using Random Forest and XGBoost
- Time-based model evaluation
- Inventory coverage analysis
- High-risk and overstock detection
- Reorder quantity recommendations
- Interactive Streamlit dashboard
- Store and product-level filtering

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit

## 📊 Machine Learning

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

## 📈 Results

| Model | MAE | RMSE |
|---|---:|---:|
| 7-Day Baseline | 93.22 | 115.46 |
| Random Forest | 90.78 | 109.82 |
| XGBoost | 89.10 | 108.73 |

## 📦 Inventory Intelligence

Inventory coverage is calculated as:

Inventory Coverage = Inventory Level / Predicted Demand

Risk categories:

- Coverage < 1 → High Risk
- 1 ≤ Coverage < 2 → Medium Risk
- 2 ≤ Coverage ≤ 3 → Low Risk
- Coverage > 3 → Overstock

The system also calculates recommended reorder quantities based
on predicted demand and inventory levels.

## 📁 Project Structure

SmartStock/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA_SmartStock.ipynb
│   └── 02_Demand_Forecasting.ipynb
│
├── models/
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md

## ▶️ Run the Dashboard

```bash
streamlit run app/app.py