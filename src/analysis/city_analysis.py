
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "fairfare_ride_demand_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
)

df = df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

numeric_features = [
    "Ride_Distance_KM",
    "Available_Drivers",
    "Demand_Score",
    "Surge_Multiplier",
    "Hour_of_Day",
    "Is_Weekend",
    "Driver_Performance_Score",
    "Driver_XP",
    "Fare_Acceptance",
    "Cancellation_Rate",
    "Traffic_Delay",
    "Cancellation_Probability",
    "Driver_Trust_Score",
    "Rider_Trust_Score",
    "Leaderboard_Rank",
]

categorical_features = [
    "City",
    "Ride_Type",
    "Weather",
    "Event",
    "Payment_Type",
    "Demand_Level",
    "Time_of_Day",
]

features = numeric_features + categorical_features
target = "Final_Fare"


# ---------------------------------------------------------
# Convert numeric columns
# ---------------------------------------------------------

for column in numeric_features + [target]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# Remove invalid rows
model_data = df[features + [target]].dropna().copy()


X = model_data[features]
y = model_data[target]


# ---------------------------------------------------------
# Chronological Train/Test Split
# ---------------------------------------------------------

split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]
)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train_processed,
    y_train,
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

predictions = model.predict(
    X_test_processed
)

mae = mean_absolute_error(
    y_test,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
    )
)

r2 = r2_score(
    y_test,
    predictions,
)


metrics_df = pd.DataFrame([
    {
        "Model": "RandomForestRegressor",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }
])


print("\nModel Evaluation:")
print(metrics_df)


metrics_df.to_csv(
    OUTPUT_DIR / "ml_model_metrics.csv",
    index=False,
)


# ---------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_,
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\nTop Feature Importances:")
print(importance_df.head(15))


importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
)


# ---------------------------------------------------------
# Prediction Results
# ---------------------------------------------------------

prediction_results = X_test.copy()

prediction_results["Actual_Fare"] = y_test.values
prediction_results["Predicted_Fare"] = predictions

prediction_results.to_csv(
    OUTPUT_DIR / "fare_predictions.csv",
    index=False,
)


print("\nPrediction Sample:")
print(
    prediction_results[
        ["Actual_Fare", "Predicted_Fare"]
    ].head(10)
)

