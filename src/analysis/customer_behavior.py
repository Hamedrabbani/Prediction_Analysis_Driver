from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "fairfare_ride_demand_dataset (1).csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Fare acceptance percentage
df["Fare_Acceptance_Percentage"] = (
    df["Fare_Acceptance"] * 100
)

city_acceptance = (
    df.groupby("City")["Fare_Acceptance_Percentage"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

city_acceptance.columns = [
    "City",
    "Average_Fare_Acceptance_Percentage",
]

print("\nFare acceptance by city:")
print(city_acceptance.head(10))

city_acceptance.to_csv(
    OUTPUT_DIR / "fare_acceptance_by_city.csv",
    index=False,
)

# Rider trust analysis
rider_trust = (
    df.groupby("City")["Rider_Trust_Score"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

rider_trust.to_csv(
    OUTPUT_DIR / "rider_trust_by_city.csv",
    index=False,
)