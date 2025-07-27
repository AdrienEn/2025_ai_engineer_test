import base64
import getpass
import json
import os
import pathlib
import time

import cv2
from langchain_core.language_models.chat_models import BaseChatModel
from openai import images

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

DATASET = "images_EST-1"


def load_images(folder_path):
    """Load images from a specified folder."""
    images_paths = [f for f in pathlib.Path(folder_path).glob("*.jpg")]
    images = []

    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        images.append(image)
        # cv2.imshow(f"Image: {image_path.name}", image)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()
    return images


def load_image(image_path):
    """Load a single image from a specified path."""
    image = cv2.imread(image_path)
    return image


def load_metadata(file_path):
    """Load metadata from a specified file."""
    with open(file_path, "r") as file:
        metadata = json.load(file)
    return metadata


def load_reports(folder_path: str) -> list[str]:
    """Load reports from a specified folder."""
    reports: list[str] = []
    report_paths = [f for f in pathlib.Path(folder_path).glob("*.md")]

    for report_path in report_paths:
        with open(report_path, "r", encoding="utf-8") as file:
            report_content = file.read()
            reports.append(report_content)

    return reports


def main():
    """Main function to run the multi-agent risk analysis system."""
    print("Hello, I'm a multi-agent risk analysis system!")

    model: BaseChatModel = init_chat_model("gpt-4o-mini", model_provider="openai")

    print("Loading metadata...")
    metadata = load_metadata(f"assets/{DATASET}.json")
    reports: list[str] = []
    print("Treating images...")
    for image in metadata["images"]:
        print(f"Treating image {image}")
        image_path = f"assets/{DATASET}/{image}"

        with open(image_path, "rb") as image_file:
            base64_bytes = base64.b64encode(image_file.read())
            base64_string = base64_bytes.decode()

        # Define prompt
        prompt = ChatPromptTemplate(
            [
                {
                    "role": "system",
                    "content": "Les rapports de risque doivent suivre le format suivant :\n\n"
                    "# Rapport d'analyse des risques sur le lieu de travail\n"
                    "## Introduction\n"
                    "Ce rapport présente une analyse des risques détectés sur le lieu de travail à partir de données multi-modales (images, détections, météo, réglementation).\n"
                    "\n"
                    "## Données utilisées\n"
                    "- Images panoramiques HD\n"
                    "- Détections de personnes\n"
                    "- Indications de risques\n"
                    "- Météo\n"
                    "- Cadre réglementaire\n"
                    "\n"
                    "## Synthèse des risques détectés\n"
                    "<!-- Résumé automatique des principaux risques identifiés -->\n"
                    "\n"
                    "## Illustration des manquements\n"
                    "<!-- Insertion d'images annotées ou de schémas -->\n"
                    "\n"
                    "## Recommandations\n"
                    "<!-- Conseils personnalisés selon la réglementation et les risques détectés -->\n"
                    "\n"
                    "## Détails\n"
                    "| Image | Risques détectés | Localisation | Recommandations |\n"
                    "|-------|------------------|--------------|-----------------|\n"
                    "|       |                  |              |                 |\n"
                    "\n"
                    "## Conclusion\n"
                    "<!-- Synthèse finale et axes d'amélioration -->\n"
                    "---\n"
                    "*Ce rapport a été généré automatiquement par le système multi-agent d'analyse des risques.*\n"
                    "",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Décris les risque apparent sur le chantier de cette image: {image_name}",
                        },
                        {
                            "type": "image",
                            "source_type": "base64",
                            "mime_type": "image/jpeg",
                            "data": "{image_data}",
                        },
                    ],
                },
            ]
        )

        chain = prompt | model
        print(f"Calling LLM")
        response = chain.invoke({"image_name": image, "image_data": base64_string})
        reports.append(str(response.content))
        with open(f"rapports/{image}.md", "w", encoding="utf-8") as report_file:
            report_file.write(str(response.content))
        time.sleep(10)  # To avoid hitting rate limits

    print("Concatenating reports...")
    prompt = ChatPromptTemplate(
        [
            {
                "role": "system",
                "content": "Les rapports de risque doivent suivre le format suivant :\n\n"
                "# Rapport d'analyse des risques sur le lieu de travail\n"
                "## Introduction\n"
                "Ce rapport présente une analyse des risques détectés sur le lieu de travail à partir de données multi-modales (images, détections, météo, réglementation).\n"
                "\n"
                "## Données utilisées\n"
                "- Images panoramiques HD\n"
                "- Détections de personnes\n"
                "- Indications de risques\n"
                "- Météo\n"
                "- Cadre réglementaire\n"
                "\n"
                "## Synthèse des risques détectés\n"
                "<!-- Résumé automatique des principaux risques identifiés -->\n"
                "\n"
                "## Illustration des manquements\n"
                "<!-- Insertion d'images annotées ou de schémas -->\n"
                "\n"
                "## Recommandations\n"
                "<!-- Conseils personnalisés selon la réglementation et les risques détectés -->\n"
                "\n"
                "## Détails\n"
                "| Image | Risques détectés | Localisation | Recommandations |\n"
                "|-------|------------------|--------------|-----------------|\n"
                "|       |                  |              |                 |\n"
                "\n"
                "## Conclusion\n"
                "<!-- Synthèse finale et axes d'amélioration -->\n"
                "---\n"
                "*Ce rapport a été généré automatiquement par le système multi-agent d'analyse des risques.*\n"
                "",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Concatène les rapports de risque suivants :",
                    },
                    {
                        "type": "text",
                        "text": "{reports}",
                    },
                ],
            },
        ]
    )

    # reports = load_reports("rapports")
    chain = prompt | model
    response = chain.invoke({"reports": "\n".join(reports)})

    with open(f"rapport_test.md", "w", encoding="utf-8") as report_file:
        report_file.write(str(response.content))


if __name__ == "__main__":
    main()
