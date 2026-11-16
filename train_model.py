from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline


DATA_FILE = Path("data/kolkata_unified_dataset.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "soil_moisture_model.joblib"
METRICS_FILE = MODEL_DIR / "model_metrics.json"

TARGET = "Volumetric_Moisture_Pct"

FEATURES = [
    "Volumetric_Moisture_Pct",
    "Sensor_Board_Temperature_C",
    "Raw_Capacitive_ADC",
    "Sensor_Voltage_V",
    "Atm_Pressure_inHg",
    "Atm_Temperature_C",
    "Soil_Temperature_C",
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
]


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"])

    required = ["Timestamp", "Farm_ID", TARGET] + FEATURES
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(
        ["Farm_ID", "Timestamp"]
    ).reset_index(drop=True)

    # Target: measured soil moisture 15 minutes later.
    df["target_next_15min"] = (
        df.groupby("Farm_ID")[TARGET].shift(-1)
    )

    next_timestamp = df.groupby("Farm_ID")["Timestamp"].shift(-1)

    valid_next = (
        next_timestamp - df["Timestamp"]
    ) == pd.Timedelta(minutes=15)

    df = df.loc[valid_next].copy()
    df = df.dropna(
        subset=FEATURES + ["target_next_15min"]
    )

    df = df.sort_values("Timestamp").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            f"Not enough valid records: {len(df)}"
        )

    # Split by timestamp so later measurements are held out.
    unique_timestamps = np.sort(df["Timestamp"].unique())
    cutoff_index = int(len(unique_timestamps) * 0.8)

    if cutoff_index <= 0 or cutoff_index >= len(unique_timestamps):
        raise ValueError("Unable to create a valid chronological split.")

    cutoff_timestamp = pd.Timestamp(unique_timestamps[cutoff_index])

    train_df = df[df["Timestamp"] < cutoff_timestamp].copy()
    test_df = df[df["Timestamp"] >= cutoff_timestamp].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing data is empty.")

    X_train = train_df[FEATURES]
    y_train = train_df["target_next_15min"]

    X_test = test_df[FEATURES]
    y_test = test_df["target_next_15min"]

    # Baseline: assume the next reading equals the current reading.
    baseline_predictions = test_df[TARGET].to_numpy()
    baseline_metrics = calculate_metrics(
        y_test, baseline_predictions
    )

    models = {
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "Linear Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            LinearRegression(),
        ),
    }

    trained_models = {}
    model_results = {}

    for name, candidate in models.items():
        print(f"\nTraining {name}...")

        candidate.fit(X_train, y_train)
        predictions = candidate.predict(X_test)

        trained_models[name] = candidate
        model_results[name] = calculate_metrics(
            y_test, predictions
        )

    # Select the ML model with the lowest test MAE.
    best_model_name = min(
        model_results,
        key=lambda name: model_results[name]["MAE"],
    )

    best_model = trained_models[best_model_name]
    best_metrics = model_results[best_model_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "features": FEATURES,
            "target": "next_15min_measured_soil_moisture_percent",
            "model_name": best_model_name,
            "training_end_timestamp": str(
                train_df["Timestamp"].max()
            ),
            "moisture_unit": "percent volumetric water content",
            "prediction_horizon": "15 minutes",
            "data_source": "EcoIrrigate dataset, Kolkata, India",
            "data_limitation": (
                "Measurements come from two Kolkata farms during "
                "January-April 2025. Generalization to other farms, "
                "locations, seasons, and sensors has not been established."
            ),
        },
        MODEL_FILE,
    )

    results = {
        "dataset": DATA_FILE.name,
        "valid_records": int(len(df)),
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "training_end_timestamp": str(
            train_df["Timestamp"].max()
        ),
        "test_start_timestamp": str(
            test_df["Timestamp"].min()
        ),
        "test_end_timestamp": str(
            test_df["Timestamp"].max()
        ),
        "target": "Measured soil moisture 15 minutes later (%)",
        "unit": "volumetric water content percent",
        "random_forest": model_results["Random Forest"],
        "linear_regression": model_results["Linear Regression"],
        "baseline": baseline_metrics,
        "best_ml_model": best_model_name,
        "best_ml_model_metrics": best_metrics,
        "warning": (
            "The evaluation uses a chronological holdout from the same "
            "dataset and does not establish performance on new farms. "
            "The baseline may outperform both ML models. These results "
            "evaluate moisture prediction, not irrigation outcomes."
        ),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 45)
    print("TRAINING AND EVALUATION COMPLETED")
    print("=" * 45)

    print(f"Valid records: {len(df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")

    for name, metrics in model_results.items():
        print(f"\n{name} metrics:")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")

    print("\nPersistence baseline metrics:")

    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.6f}")

    print(f"\nBest ML model by test MAE: {best_model_name}")

    if best_metrics["MAE"] < baseline_metrics["MAE"]:
        print("The selected ML model outperformed the baseline on MAE.")
    else:
        print(
            "WARNING: Neither ML model was selected over the baseline "
            "based on MAE. Do not claim that ML performed better."
        )

    print(f"\nModel saved to: {MODEL_FILE.resolve()}")
    print(f"Metrics saved to: {METRICS_FILE.resolve()}")

    print(
        "\nNote: This model predicts measured soil moisture 15 minutes "
        "ahead. It does not directly predict irrigation requirements."
    )


if __name__ == "__main__":
    main()