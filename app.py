import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="E-Commerce Delivery Prediction",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 E-Commerce Delivery Time Prediction")
st.write(
    "Predict estimated delivery time using Machine Learning based on "
    "distance, packages, traffic, weather, and delivery type."
)

st.divider()


# -----------------------------
# Dataset
# -----------------------------
data = {
    "Distance_km": [2, 5, 8, 10, 12, 15, 18, 20, 7, 14, 6, 11],
    "Packages": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 3, 4],
    "Traffic": [1, 2, 3, 2, 3, 4, 4, 5, 2, 3, 1, 4],
    "Weather": [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 1, 3],
    "DeliveryType": [1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2],
    "DeliveryTime_hr": [
        1.0, 2.0, 3.2, 4.0, 4.8, 6.0,
        7.2, 8.0, 2.5, 5.5, 2.8, 4.5
    ]
}

df = pd.DataFrame(data)


# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("DeliveryTime_hr", axis=1)
y = df["DeliveryTime_hr"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Train Models
# -----------------------------
lr_model = LinearRegression()

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_split=4,
    random_state=42
)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    return mae, rmse, r2


lr_mae, lr_rmse, lr_r2 = evaluate_model(y_test, lr_pred)
rf_mae, rf_rmse, rf_r2 = evaluate_model(y_test, rf_pred)


# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📦 Order Details")

distance = st.sidebar.number_input(
    "Distance (km)",
    min_value=1.0,
    max_value=100.0,
    value=10.0,
    step=1.0
)

packages = st.sidebar.number_input(
    "Number of Packages",
    min_value=1,
    max_value=50,
    value=3,
    step=1
)

traffic = st.sidebar.slider(
    "Traffic Level",
    min_value=1,
    max_value=5,
    value=3,
    help="1 = Low, 5 = Very High"
)

weather = st.sidebar.slider(
    "Weather Condition",
    min_value=1,
    max_value=4,
    value=2,
    help="1 = Clear, 4 = Storm"
)

delivery_type = st.sidebar.selectbox(
    "Delivery Type",
    ["Standard", "Express"]
)

delivery_code = 1 if delivery_type == "Standard" else 2


# -----------------------------
# Prediction
# -----------------------------
new_order = pd.DataFrame(
    [[distance, packages, traffic, weather, delivery_code]],
    columns=[
        "Distance_km",
        "Packages",
        "Traffic",
        "Weather",
        "DeliveryType"
    ]
)

lr_prediction = lr_model.predict(new_order)[0]
rf_prediction = rf_model.predict(new_order)[0]


# -----------------------------
# Prediction Results
# -----------------------------
st.subheader("📊 Delivery Prediction")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Linear Regression",
        f"{lr_prediction:.2f} hours"
    )

with col2:
    st.metric(
        "Random Forest",
        f"{rf_prediction:.2f} hours"
    )


st.info(
    f"Estimated delivery time ranges from "
    f"{min(lr_prediction, rf_prediction):.2f} to "
    f"{max(lr_prediction, rf_prediction):.2f} hours."
)


# -----------------------------
# Model Performance
# -----------------------------
st.divider()

st.subheader("📈 Model Performance")

performance_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        lr_mae,
        rf_mae
    ],
    "RMSE": [
        lr_rmse,
        rf_rmse
    ],
    "R² Score": [
        lr_r2,
        rf_r2
    ]
})

st.dataframe(
    performance_df.style.format({
        "MAE": "{:.3f}",
        "RMSE": "{:.3f}",
        "R² Score": "{:.3f}"
    }),
    use_container_width=True
)


# -----------------------------
# Actual vs Predicted
# -----------------------------
st.divider()

st.subheader("🎯 Actual vs Predicted Delivery Time")

fig, ax = plt.subplots()

ax.scatter(
    y_test,
    lr_pred,
    label="Linear Regression"
)

ax.scatter(
    y_test,
    rf_pred,
    label="Random Forest"
)

ax.set_xlabel("Actual Delivery Time (hours)")
ax.set_ylabel("Predicted Delivery Time (hours)")
ax.set_title("Actual vs Predicted Delivery Time")
ax.legend()

st.pyplot(fig)


# -----------------------------
# Dataset
# -----------------------------
with st.expander("🔍 View Training Dataset"):
    st.dataframe(
        df,
        use_container_width=True
    )


# -----------------------------
# Project Information
# -----------------------------
st.divider()

st.caption(
    "Machine Learning Project | Python | Pandas | NumPy | "
    "Scikit-learn | Matplotlib | Streamlit"
)
