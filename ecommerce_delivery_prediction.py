import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    print("\n--- Advanced E-Commerce Delivery Time Prediction ---\n")

    # Expanded Dataset
    data = {
        "Distance_km": [2, 5, 8, 10, 12, 15, 18, 20, 7, 14, 6, 11],
        "Packages": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 3, 4],
        "Traffic": [1, 2, 3, 2, 3, 4, 4, 5, 2, 3, 1, 4],  # 1=Low, 5=High
        "Weather": [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 1, 3],  # 1=Clear, 4=Storm
        "DeliveryType": [1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2],  # 1=Standard, 2=Express
        "DeliveryTime_hr": [1.0, 2.0, 3.2, 4.0, 4.8, 6.0, 7.2, 8.0, 2.5, 5.5, 2.8, 4.5]
    }

    df = pd.DataFrame(data)
    print(df)

    # Features & Target
    X = df.drop("DeliveryTime_hr", axis=1)
    y = df["DeliveryTime_hr"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_split=4,
    random_state=42
    )

    # Train
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Evaluation function
    def evaluate(name, y_test, pred):
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        print(f"\n{name} Performance:")
        print("MAE :", round(mae, 2))
        print("RMSE:", round(rmse, 2))
        print("R²  :", round(r2, 2))

    evaluate("Linear Regression", y_test, lr_pred)
    evaluate("Random Forest", y_test, rf_pred)

    # Visualization
    plt.figure()
    plt.scatter(y_test, lr_pred)
    plt.xlabel("Actual Delivery Time")
    plt.ylabel("Predicted Delivery Time")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.show()

    plt.figure()
    plt.scatter(y_test, rf_pred)
    plt.xlabel("Actual Delivery Time")
    plt.ylabel("Predicted Delivery Time")
    plt.title("Random Forest: Actual vs Predicted")
    plt.show()

    # New Prediction
    new_order = pd.DataFrame([[10, 3, 3, 2, 1]],
                             columns=["Distance_km", "Packages", "Traffic", "Weather", "DeliveryType"])

    lr_time = lr_model.predict(new_order)
    rf_time = rf_model.predict(new_order)

    print("\nNew Order Prediction:")
    print("Linear Regression:", round(lr_time[0], 2), "hours")
    print("Random Forest   :", round(rf_time[0], 2), "hours")


if __name__ == "__main__":
    main()