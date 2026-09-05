
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "fairfare_ride_demand_dataset (1).csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
)

all_predictions = []
metrics = []

for city, df_city in df.groupby("City"):

    # Aggregate ride-level data into daily city-level average fare
    daily_data = (
        df_city.groupby("Date", as_index=False)["Final_Fare"]
        .mean()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Create elapsed-day index
    daily_data["Day_Index"] = (
        daily_data["Date"] - daily_data["Date"].min()
    ).dt.days

    if len(daily_data) < 10:
        continue

    X = daily_data[["Day_Index"]]
    y = daily_data["Final_Fare"]

    # Chronological 80/20 train-test split
    split_index = int(len(daily_data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    metrics.append({
        "City": city,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": rmse,
        "R2": r2_score(y_test, predictions),
    })

    # Forecast the next 9 days
    last_day_index = daily_data["Day_Index"].max()

    future_indexes = np.arange(
        last_day_index + 1,
        last_day_index + 10,
    )

    # Use DataFrame to preserve feature names
    future_X = pd.DataFrame({
        "Day_Index": future_indexes
    })

    future_predictions = model.predict(future_X)

    last_date = daily_data["Date"].max()

    for day_index, prediction in zip(
        future_indexes,
        future_predictions,
    ):
        future_date = (
            last_date
            + pd.Timedelta(
                days=int(day_index - last_day_index)
            )
        )

        all_predictions.append({
            "City": city,
            "Date": future_date.strftime("%Y-%m-%d"),
            "Day_Index": day_index,
            "Predicted_Average_Daily_Fare": round(
                float(prediction),
                2,
            ),
        })


predictions_df = pd.DataFrame(all_predictions)
metrics_df = pd.DataFrame(metrics)

predictions_df.to_csv(
    OUTPUT_DIR / "future_fare_predictions.csv",
    index=False,
)

metrics_df.to_csv(
    OUTPUT_DIR / "forecasting_metrics.csv",
    index=False,
)

print("\nForecasting metrics:")
print(metrics_df)

print("\nFuture fare predictions:")
print(predictions_df.head(20))

