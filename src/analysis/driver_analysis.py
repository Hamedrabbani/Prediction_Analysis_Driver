from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "fairfare_ride_demand_dataset (1).csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Driver performance
driver_performance = (
    df.groupby("City")[
        ["Driver_Performance_Score", "Driver_XP", "Driver_Trust_Score"]
    ]
    .mean()
    .sort_values("Driver_Performance_Score", ascending=False)
    .reset_index()
)

print("\nDriver performance by city:")
print(driver_performance.head(10))

driver_performance.to_csv(
    OUTPUT_DIR / "driver_performance_by_city.csv",
    index=False,
)

# Distance analysis
distance_analysis = (
    df.groupby("City")["Ride_Distance_KM"]
    .agg(["sum", "mean", "max", "min"])
    .sort_values("sum", ascending=False)
    .reset_index()
)

distance_analysis.columns = [
    "City",
    "Total_Distance_KM",
    "Average_Distance_KM",
    "Maximum_Distance_KM",
    "Minimum_Distance_KM",
]

print("\nDistance analysis:")
print(distance_analysis.head(10))

distance_analysis.to_csv(
    OUTPUT_DIR / "distance_analysis_by_city.csv",
    index=False,
)

# Long-distance rides
long_distance_rides = df[
    df["Ride_Distance_KM"] > 20
][
    ["City", "Ride_Distance_KM", "Final_Fare", "Driver_Performance_Score"]
]

long_distance_rides.to_csv(
    OUTPUT_DIR / "long_distance_rides.csv",
    index=False,
)