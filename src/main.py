import concurrent.futures
import json
from typing import Dict

from camera_tool import analyze_camera
from summary import generate_final_report
from meteo_tool import summarize_weather_conditions


def load_config(path: str) -> dict:
    """
    Load a configuration from a JSON file.

    Args:
        path (str): Path to the JSON configuration file.

    Returns:
        dict: Dictionary representing the JSON content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {path}")
        raise e
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format in file: {path}")
        raise e


def analyze_all_cameras(config: dict) -> Dict[str, str]:
    """
    Analyze all cameras in parallel using the provided configuration.

    Args:
        config (List[Dict[str, Any]]): A list of configuration dictionaries,
            each containing keys such as:
            - "name": str, name of the camera
            - "image_dir": str, path to the directory containing images
            - "detections_json": str, path to the JSON file with object detections

    Returns:
        Dict[str, str]: A dictionary mapping each camera name to the result of its analysis.
        If an error occurs during the analysis of a camera, the value will be an error message.
    """
    results: Dict[str, str] = {}

    # Parallel analysis for each camera
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                analyze_camera,
                camera_name=camera["name"],
                images_folder=camera["image_dir"],
                json_path=camera["detections_json"],
            ): camera["name"]
            for camera in config
        }

        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"Error while analyzing {name}: {e}")
                results[name] = f"Error for {name}"

    return results


if __name__ == "__main__":
    print("Multi-agent system analyzing the risks – Starting...")

    # Load system configuration (camera paths, detection files, etc.)
    config: dict = load_config("../config/config.json")
    # Analyze camera's data in parallel (image + detection JSON)
    camera_summaries = analyze_all_cameras(config)
    # Generate a text summary of weather conditions
    weather_summary = summarize_weather_conditions("../assets/weather_info.json")

    # Here we retrieve only the images from EST-1
    middle_camera_summary = camera_summaries.get("Camera_Milieu", "No data available")

    report = generate_final_report(middle_camera_summary, weather_summary)

