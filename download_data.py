import requests
import pandas as pd
from pathlib import Path

LATITUDE = 31.22
LONGITUDE = 75.77

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

OUTPUT_FILE = Path("data") / "historical_soil_weather.csv"


def download_weather_data():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "et0_fao_evapotranspiration"
        ],
        "timezone": "Asia/Kolkata"
    }

    print("Connecting to the historical weather service...")

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "daily" not in data:
        raise ValueError("The weather service returned no daily data.")

    daily = data["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "temperature_c": daily["temperature_2m_mean"],
        "humidity_percent": daily["relative_humidity_2m_mean"],
        "precipitation_mm": daily["precipitation_sum"],
        "et0_mm": daily["et0_fao_evapotranspiration"]
    })

    df["date"] = pd.to_datetime(df["date"])

    df = df.dropna().sort_values("date").reset_index(drop=True)

    # These are approximate moisture indicators derived from weather.
    # They are NOT direct measurements from soil sensors.
    df["soil_moisture_shallow"] = 0.35
    df["soil_moisture_deeper"] = 0.40

    df.loc[
        df["precipitation_mm"] > 5,
        "soil_moisture_shallow"
    ] += 0.05

    df["soil_moisture_shallow"] -= (
        df["et0_mm"] * 0.005
    )

    df["soil_moisture_shallow"] = (
        df["soil_moisture_shallow"].clip(0.10, 0.60)
    )

    df["soil_moisture_deeper"] = (
        df["soil_moisture_deeper"].clip(0.10, 0.60)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDownload completed successfully!")
    print(f"Records saved: {len(df)}")
    print(f"Saved file: {OUTPUT_FILE.resolve()}")
    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print(
        "\nIMPORTANT: Weather fields come from the weather service. "
        "Soil-moisture columns are approximate synthetic indicators, "
        "not actual sensor observations."
    )


if __name__ == "__main__":
    try:
        download_weather_data()
    except Exception as error:
        print(f"\nDownload failed: {error}")
        print("Check your internet connection and try again.")