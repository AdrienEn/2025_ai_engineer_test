import json
from datetime import datetime
from typing import Optional


def summarize_weather_conditions(
    weather_json_path: str, hour: Optional[str] = None
) -> str:
    """
    Extracts and summarizes key weather and air quality conditions from a JSON file.

    Args:
        weather_json_path (str): Path to the JSON file containing weather and air quality data.
        hour (Optional[str]): Specific hour to extract data for (in ISO 8601 format).
                              If not provided, uses the most recent available hour.

    Returns:
        str: Formatted string summarizing the weather and air quality conditions.
    """

    # Load weather and air quality data
    with open(weather_json_path, encoding="utf-8") as f:
        meteo = json.load(f)["weather_infos"]["1374225"]

    # Get available hourly time slots
    hours = meteo["forecast"]["hourly"]["time"]

    # If no hour is provided, use the latest
    idx = -1 if hour is None else hours.index(hour)

    # Extract relevant sections
    forecast = meteo["forecast"]["hourly"]
    air = meteo["airquality"]["hourly"]
    aqi = meteo["airquality.forecast"]["hourly"]

    # Helper to safely get values from a list by index
    def get_val(source_dict, key):
        return source_dict.get(key, [None])[idx]

    # Construct summary text with relevant metrics
    summary = f"""
    Weather conditions at {hours[idx]}:
    - Temperature: {get_val(forecast, 'temperature_2m')} °C
    - Humidity: {get_val(forecast, 'relativehumidity_2m')}%
    - Wind Speed: {get_val(forecast, 'windspeed_10m')} km/h
    - Precipitation: {get_val(forecast, 'precipitation')} mm
    - Cloud Cover: {get_val(forecast, 'cloudcover')}%
    - Air Quality Index (EAQI): {get_val(aqi, 'european_aqi')}
    - PM2.5 (Fine particles): {get_val(air, 'pm2_5')} µg/m³
    - Ozone (O₃): {get_val(air, 'ozone')} µg/m³
    """.strip()

    return summary
