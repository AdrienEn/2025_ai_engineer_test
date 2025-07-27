import json
from pathlib import Path
from typing import List

import cv2

DATASET = "images_EST-1"
THRESHOLD = 0.5


def load_images(folder_path: str) -> List[cv2.typing.MatLike]:
    """Load images from a specified folder."""
    images_paths: list[Path] = [f for f in Path(folder_path).glob("*.jpg")]
    images: List[cv2.typing.MatLike] = []

    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        images.append(image)

    return images


def load_image(image_path: str) -> cv2.typing.MatLike | None:
    """Load a single image from a specified path."""
    image = cv2.imread(image_path)
    return image


def load_metadata(file_path: str) -> dict | None:
    """Load metadata from a specified file."""
    with open(file_path, "r") as file:
        metadata = json.load(file)
    return metadata


def main() -> None:
    """Main function to run the multi-agent risk analysis system."""
    print("Hello, I'm a multi-agent risk analysis system!")

    metadata = load_metadata(f"assets/{DATASET}.json")
    if metadata is None:
        print("Failed to load metadata.")
        return

    for image in metadata["images"]:
        image_path = f"assets/{DATASET}/{image}"
        loaded_image = load_image(image_path)
        if loaded_image is None:
            print(f"Error loading image: {image}")
            continue
        height, width, _ = loaded_image.shape
        print(f"Loaded image: {image} with dimensions: {width}x{height}")
        for detection in metadata["images"][image]["detections"]:
            if detection["score"] < THRESHOLD:
                continue
            cv2.rectangle(
                loaded_image,
                (
                    int(width * detection["bounding_box_start_x"]),
                    int(height * detection["bounding_box_start_y"]),
                ),
                (
                    int(width * detection["bounding_box_end_x"]),
                    int(height * detection["bounding_box_end_y"]),
                ),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                loaded_image,
                str(detection["score"]),
                (
                    int(width * detection["bounding_box_start_x"]),
                    int(height * detection["bounding_box_start_y"]) - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                loaded_image,
                str(detection["label"]),
                (
                    int(width * detection["bounding_box_start_x"]),
                    int(height * detection["bounding_box_start_y"]) - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        cv2.imshow(f"Image", loaded_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
