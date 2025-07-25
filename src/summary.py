import os
from typing import Optional
from langchain_ollama import ChatOllama


def generate_summary(
    middle_camera_summary: str,
    weather_summary: str,
    entry_camera_summary: Optional[str] = None,
) -> str:
    """
    Generates a global risk summary using image analysis from one or two cameras
    and weather conditions, leveraging a local LLM.

    Args:
        middle_camera_summary (str): Summary of risk analysis from the middle camera.
        weather_summary (str): Summary of weather and environmental conditions.
        entry_camera_summary (Optional[str]): Optional summary from the entry camera.
                                              If not provided, only the middle camera is considered.

    Returns:
        str: A concise global risk summary, possibly including recommendations or confirming the absence of risks.
    """
    if entry_camera_summary is None:
        prompt = f"""
        Voici les analyses d'images de cameras sur un chantier.
        === Camera_Milieu ===
        {middle_camera_summary}

        === Conditions météo ===
        {weather_summary}

        Dresse une synthese globale des risques detectes. , en prenant aussi en compte les conditions meteo si elles peuvent aggraver un risque ou en creer un nouveau.
        Si aucun n'est detecte, conclus par :'Aucun risque detecte pour l'instant, surveillance continue recommandee.'
        """
    else:
        prompt = f"""
        Voici les analyses de deux cameras sur un chantier.
        === Camera_Milieu ===
        {middle_camera_summary}

        === Camera_Entree ===
        {entry_camera_summary}

        === Conditions météo ===
        {weather_summary}

        Dresse une synthese globale des risques detectes. , en prenant aussi en compte les conditions meteo si elles peuvent aggraver un risque ou en creer un nouveau.
        Si aucun n'est detecte, conclus par :'Aucun risque detecte pour l'instant, surveillance continue recommandee.'
        """
    llm = ChatOllama(model="llama3:8b", temperature=0.3)
    return llm.invoke(prompt).content


def generate_final_report(
    middle_camera_summary: str,
    weather_summary: str,
    entry_camera_summary: Optional[str] = None,
    annotated_images_dir: str = "../assets/annotated_images",
) -> str:
    """
    Generates the final risk analysis report as a Markdown file by compiling
    summaries from camera analyses, weather conditions, and annotated images.

    Args:
        middle_camera_summary (str): Summary of risk analysis from the middle camera.
        weather_summary (str): Summary of weather and environmental conditions.
        entry_camera_summary (Optional[str]): Optional summary from the entry camera.
        annotated_images_dir (str): Path to the directory containing annotated images.

    Returns:
        str: A success message with the path to the generated report.
    """
    synthese = generate_summary(
        middle_camera_summary=middle_camera_summary,
        weather_summary=weather_summary,
        entry_camera_summary=entry_camera_summary,
    )

    # Recherche des images annotées
    annotated_images = []
    for root, _, files in os.walk(annotated_images_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                relative_path = os.path.join(root, f)
                annotated_images.append(relative_path)

    # Construction du rapport
    report_path = "rapport_final.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            f"""# Rapport d'analyse des risques sur le lieu de travail

## Introduction
Ce rapport présente une analyse des risques détectés sur le lieu de travail à partir de données multi-modales (images, détections, météo, réglementation).

## Données utilisées
- Images panoramiques HD
- Détections de personnes
- Indications de risques 
- Météo
- Cadre réglementaire

## Synthèse des risques détectés
{synthese}

## Illustration des manquements
{chr(10).join(f"- ![]({img})" for img in annotated_images) if annotated_images else "Aucune illustration disponible."}

## Recommandations
À partir des détections et du cadre réglementaire, il est recommandé d'examiner de près les zones signalées, notamment celles contenant des comportements à risque. Une vigilance accrue est aussi recommandée en cas de conditions météo défavorables.
=== Camera_Milieu ===
{middle_camera_summary}

=== Camera_Entree ===
{entry_camera_summary}

## Conclusion
Ce rapport met en évidence les risques potentiels détectés automatiquement. Il est conseillé de procéder à une évaluation humaine pour compléter cette analyse.

---
*Ce rapport a été généré automatiquement par le système multi-agent d'analyse des risques.*
"""
        )
    return f"✅ Rapport final généré : {report_path}"
