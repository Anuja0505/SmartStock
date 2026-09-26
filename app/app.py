import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="SmartStock",
    page_icon="📦",
    layout="wide"
)

st.title("📦 SmartStock")
st.markdown(
    "**Demand Forecasting & Inventory Risk Intelligence**"
)

st.caption(
    "Forecast demand • Detect inventory risk • Recommend reorder quantities"
)

st.divider()

st.info(
    "SmartStock uses historical sales patterns and machine learning "
    "to forecast product demand, assess inventory coverage, and "
    "generate reorder recommendations."
)

st.write(
    "Forecast demand, identify inventory risk, "
    "and generate reorder recommendations."
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


@st.cache_data
def load_data():
    inventory_summary = pd.read_csv(
        DATA_DIR / "inventory_summary.csv"
    )

    test_predictions = pd.read_csv(
        DATA_DIR / "test_predictions.csv"
    )

    feature_importance = pd.read_csv(
        DATA_DIR / "feature_importance.csv"
    )
    test_predictions["Date"] = pd.to_datetime(
        test_predictions["Date"]
    )

    return inventory_summary, test_predictions, feature_importance

inventory_summary, test_predictions, feature_importance = load_data()

st.markdown("### Inventory Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Store-Product Combinations",
    len(inventory_summary)
)

col2.metric(
    "High-Risk Days",
    int(inventory_summary["High_Risk_Days"].sum())
)

col3.metric(
    "Total Reorder Quantity",
    int(inventory_summary["Total_Reorder"].sum())
)

col4.metric(
    "Overstock Days",
    int(inventory_summary["Overstock_Days"].sum())
)

st.markdown("### Inventory Risk Distribution")

# Calculate inventory coverage
test_predictions["Inventory Coverage"] = (
    test_predictions["Inventory Level"] /
    test_predictions["Predicted Demand"]
)

# Classify inventory risk
test_predictions["Risk Level"] = pd.cut(
    test_predictions["Inventory Coverage"],
    bins=[-float("inf"), 1, 2, 3, float("inf")],
    labels=[
        "High Risk",
        "Medium Risk",
        "Low Risk",
        "Overstock"
    ],
    right=False
)

risk_counts = (
    test_predictions["Risk Level"]
    .value_counts()
    .reindex(
        ["High Risk", "Medium Risk", "Low Risk", "Overstock"],
        fill_value=0
    )
)

st.bar_chart(risk_counts)

st.markdown("### Filter Inventory")

col1, col2 = st.columns(2)

with col1:
    selected_store = st.selectbox(
        "Select Store",
        ["All"] + sorted(test_predictions["Store ID"].unique().tolist())
    )

with col2:
    selected_product = st.selectbox(
        "Select Product",
        ["All"] + sorted(test_predictions["Product ID"].unique().tolist())
    )

filtered_data = test_predictions.copy()

if selected_store != "All":
    filtered_data = filtered_data[
        filtered_data["Store ID"] == selected_store
    ]

if selected_product != "All":
    filtered_data = filtered_data[
        filtered_data["Product ID"] == selected_product
    ]

st.markdown("### Inventory Details")

display_columns = [
    "Date",
    "Store ID",
    "Product ID",
    "Units Sold",
    "Predicted Demand",
    "Inventory Level",
    "Inventory Coverage",
    "Risk Level"
]

st.dataframe(
    filtered_data[display_columns].sort_values("Date", ascending=False),
    use_container_width=True
)

st.markdown("### Reorder Recommendation")

filtered_data["Recommended Reorder"] = (
    filtered_data["Predicted Demand"] -
    filtered_data["Inventory Level"]
).clip(lower=0).apply(lambda x: int(np.ceil(x)))

reorder_data = filtered_data[
    filtered_data["Recommended Reorder"] > 0
].copy()

reorder_columns = [
    "Date",
    "Store ID",
    "Product ID",
    "Predicted Demand",
    "Inventory Level",
    "Risk Level",
    "Recommended Reorder"
]

st.dataframe(
    reorder_data[reorder_columns].sort_values(
        "Recommended Reorder",
        ascending=False
    ),
    use_container_width=True
)

st.markdown("### Demand Forecast")

if len(filtered_data) > 0:

    chart_data = (
        filtered_data
        .sort_values("Date")
        .set_index("Date")[[
            "Units Sold",
            "Predicted Demand"
        ]]
    )

    st.line_chart(chart_data)

else:
    st.info("No data available for the selected filters.")

st.markdown("### 🚨 Urgent Inventory Alerts")

high_risk = filtered_data[
    filtered_data["Risk Level"] == "High Risk"
].copy()

if len(high_risk) > 0:

    high_risk = high_risk.sort_values(
        "Inventory Coverage"
    )

    alert_columns = [
        "Date",
        "Store ID",
        "Product ID",
        "Units Sold",
        "Predicted Demand",
        "Inventory Level",
        "Inventory Coverage",
        "Risk Level"
    ]

    st.dataframe(
        high_risk[alert_columns].head(20),
        use_container_width=True
    )

else:
    st.success("No high-risk inventory detected for the selected filters.")


st.markdown("### 📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Baseline MAE",
    "93.22"
)

col2.metric(
    "XGBoost MAE",
    "89.10"
)

col3.metric(
    "XGBoost RMSE",
    "108.73"
)

st.caption(
    "XGBoost was evaluated using a time-based test set. "
    "Lower MAE and RMSE indicate better forecasting accuracy."
)

st.markdown("### 🧠 XGBoost Feature Importance")

st.caption(
    "Feature importance shows which features contributed most "
    "to the XGBoost model's predictions."
)

importance_chart = (
    feature_importance
    .sort_values("Importance")
    .set_index("Feature")
)

st.bar_chart(
    importance_chart
)

st.markdown(
    """
    **How to interpret this:**  
    The model uses both historical demand patterns and calendar
    features when generating forecasts. Rolling demand features,
    lag features, and seasonal timing variables all contribute
    to the prediction.
    """
)