# Driver Fare & Demand Analysis

A Python-based data analysis and machine learning project for analyzing ride-hailing data, driver performance, city-level fare patterns, and short-term fare forecasting.

## Overview

This project analyzes ride-hailing data across multiple cities to identify patterns related to:

* Ride fares
* Driver performance
* Driver experience and trust
* Ride distance
* Fare acceptance
* Demand
* Driver availability
* Traffic conditions
* Short-term fare forecasting

The project combines **Data Analysis**, **Exploratory Analysis**, and a simple **Machine Learning forecasting baseline**.

## Dataset

The dataset contains ride-level information with features including:

* `City`
* `Date`
* `Day_of_Week`
* `Ride_Distance_KM`
* `Ride_Type`
* `Weather`
* `Event`
* `Payment_Type`
* `Available_Drivers`
* `Demand_Level`
* `Surge_Multiplier`
* `Final_Fare`
* `Hour_of_Day`
* `Is_Weekend`
* `Driver_Performance_Score`
* `Driver_XP`
* `Fare_Acceptance`
* `Cancellation_Rate`
* `Demand_Score`
* `Driver_Availability`
* `Traffic_Delay`
* `Cancellation_Probability`
* `Driver_Trust_Score`
* `Rider_Trust_Score`
* `Leaderboard_Rank`

## Project Structure

```text
Prediction_Analysis_Driver/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── fairfare_ride_demand_dataset.csv
│
├── src/
│   ├── analysis/
│   │   ├── city_analysis.py
│   │   ├── customer_behavior.py
│   │   └── driver_analysis.py
│   │
│   └── forecasting/
│       └── fare_forecasting.py
│
├── outputs/
│
└── notebooks/
```

## Analysis Modules

### City Analysis

`city_analysis.py` analyzes fare distribution across cities and calculates:

* Total fare
* Average fare
* Maximum fare

Results are saved to:

```text
outputs/city_fare_analysis.csv
```

### Customer Behavior

`customer_behavior.py` analyzes:

* Fare acceptance
* Fare acceptance percentage
* Rider trust score

Results are saved to:

```text
outputs/fare_acceptance_by_city.csv
outputs/rider_trust_by_city.csv
```

### Driver Analysis

`driver_analysis.py` evaluates:

* Driver performance
* Driver experience
* Driver trust
* Ride distance
* Long-distance rides

Results are saved to:

```text
outputs/driver_performance_by_city.csv
outputs/distance_analysis_by_city.csv
outputs/long_distance_rides.csv
```

## Fare Forecasting

`fare_forecasting.py` provides a baseline forecasting model using **Linear Regression**.

For each city:

1. Data is sorted chronologically.
2. A time-based `Day_Index` is created.
3. The first 80% of observations are used for training.
4. The remaining 20% are used for evaluation.
5. Model performance is evaluated using:

   * MAE
   * RMSE
   * R²
6. Fare is forecast for the next 9 days.

Generated files:

```text
outputs/future_fare_predictions.csv
outputs/forecasting_metrics.csv
```

## Evaluation

The forecasting model is treated as a **baseline**, rather than a production forecasting solution.

The evaluation uses a chronological train/test split to avoid random shuffling of time-dependent observations.

Future improvements can compare this baseline against stronger time-series and machine-learning approaches.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## Installation

```bash
git clone https://github.com/HamedRabbani/Prediction_Analysis_Driver.git

cd Prediction_Analysis_Driver

pip install -r requirements.txt
```

## Usage

Run the city analysis:

```bash
python src/analysis/city_analysis.py
```

Run customer behavior analysis:

```bash
python src/analysis/customer_behavior.py
```

Run driver analysis:

```bash
python src/analysis/driver_analysis.py
```

Run fare forecasting:

```bash
python src/forecasting/fare_forecasting.py
```

Generated analytical results will be stored in the `outputs/` directory.

## Future Improvements

* Exploratory Data Analysis and visualization
* Feature engineering
* Stronger forecasting models
* Cross-validation for time-dependent data
* Demand forecasting
* Driver-level performance modeling
* Cancellation prediction
* Model comparison and experiment tracking
* Automated testing
* Data validation pipeline

## Author

**Hamed Rabbani**

Data Science | Machine Learning | AI Engineering
