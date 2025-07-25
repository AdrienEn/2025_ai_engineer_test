import os
import base64
import json
from typing import List, Dict, Any
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from unidecode import unidecode
from concurrent.futures import ThreadPoolExecutor, as_completed


def encode_image(image_path: str) -> str:
    """
    Reads an image file and returns its content encoded as a base64 string.

    Args:
        image_path (str): The path to the image file to encode.

    Returns:
        str: The base64-encoded string representation of the image.
    """
    # Open the image file in binary mode and read its contents
    with open(image_path, "rb") as img_file:
        # Encode the binary data to base64 and decode to UTF-8 string
        return base64.b64encode(img_file.read()).decode("utf-8")


def contains_risk_text(text: str, keywords: List[str]) -> bool:
    """
    Checks if any of the specified keywords are present in the given text,
    after normalizing the text to ASCII and converting to lowercase.

    Args:
        text (str): The text to analyze. It can be a string or an object with a 'content' attribute.
        keywords (List[str]): A list of keywords to search for in the text.

    Returns:
        bool: True if any keyword is found in the normalized text, False otherwise.
    """
    # Extract text content if it has a 'content' attribute, otherwise use text directly
    normalized_text = unidecode(
        text.content.lower() if hasattr(text, "content") else text.lower()
    )

    # Check if any keyword exists in the normalized text
    return any(keyword in normalized_text for keyword in keywords)


def annotate_image(
    image_path: str, objects: List[Dict[str, Any]], output_folder: str
) -> str:
    """
    Annotates the given image with bounding boxes and labels for detected objects,
    then saves the annotated image to the specified output folder.

    Args:
        image_path (str): Path to the input image file.
        objects (List[Dict[str, Any]]): List of detected objects, each represented by a dictionary
            containing bounding box coordinates and labels. Expected keys:
            - 'bounding_box_start_x', 'bounding_box_start_y': top-left coordinates (normalized 0-1)
            - 'bounding_box_end_x', 'bounding_box_end_y': bottom-right coordinates (normalized 0-1)
            - 'label': object label (string)
            - 'score': confidence score (float)
            - optional 'risque': risk label (string)
        output_folder (str): Directory path where the annotated image will be saved.

    Returns:
        str: Path to the saved annotated image.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open image and convert to RGB
    image: Image.Image = Image.open(image_path).convert("RGB")

    # Resize image to fit within a 960x346 box, maintaining aspect ratio
    image.thumbnail((960, 346), Image.LANCZOS)

    draw = ImageDraw.Draw(image)

    # Try loading Arial font, fallback to default font on failure
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Draw bounding boxes and labels for each detected object
    for obj in objects:
        x1 = obj.get("bounding_box_start_x", 0) * 960
        y1 = obj.get("bounding_box_start_y", 0) * 346
        x2 = obj.get("bounding_box_end_x", 0) * 960
        y2 = obj.get("bounding_box_end_y", 0) * 346

        label = f"{obj.get('label', 'unknown')} ({obj.get('score', 0):.2f})"
        if "risque" in obj:
            label += f" [{obj['risque']}]"

        # Draw rectangle box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Construct output path and save the annotated image
    output_path: str = os.path.join(output_folder, os.path.basename(image_path))
    image.save(output_path, quality=70, optimize=True)

    return output_path


def associate_risks(
    objects: List[Dict[str, Any]], other: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Associates risk labels from an additional metadata dictionary to person objects
    within a list of detected objects.

    Args:
        objects (List[Dict[str, Any]]): List of detected objects, each represented as a dictionary.
        other (Dict[str, Any]): Additional metadata dictionary which may contain risk information.
            Expected keys:
            - 'category' (str): Category name, e.g., "Risque"
            - 'label' (str): Risk label to associate, if any.

    Returns:
        List[Dict[str, Any]]: The updated list of objects with risk labels assigned to persons if applicable.
    """
    # If either input is empty, return objects unchanged
    if not other or not objects:
        return objects

    # Check if 'other' indicates a risk category with a valid label
    if other.get("category") == "Risque" and other.get("label"):
        # Assign the risk label to all objects labeled as "person"
        for obj in objects:
            if obj.get("label") == "person":
                obj["risque"] = other["label"]

    return objects


def create_regulatory_tool() -> Tool:
    """
    Creates a regulatory compliance tool using a language model prompt focused on construction site safety.

    Returns:
        Tool: A LangChain Tool instance wrapping the regulatory compliance function.
    """

    # Define the prompt template in French as required
    prompt = ChatPromptTemplate.from_template(
        """
Tu es un expert en sécurité sur les chantiers BTP, spécialisé dans la conformité réglementaire (code du travail, INRS, OPPBTP...).

Voici la situation observée :
{description}

Analyse s'il y a un non-respect d'une règle de sécurité, et cite les articles ou normes concernées si possible.
Sois précis, utilise les textes français en vigueur.
Ignore les éléments non liés à la sécurité. Sois bref et concis. Ne me dis pas bonjour.
"""
    )

    # Initialize the language model with low temperature for precise answers
    llm = ChatOllama(model="llama3:8b", temperature=0.2)

    # Compose the prompt and the LLM into a callable chain
    chain = prompt | llm

    def regulatory_advisor(description: str) -> str:
        return chain.invoke({"description": description}).content.strip()

    # Return the tool wrapping the internal function
    return Tool(
        name="ConseillerReglementaire",
        func=regulatory_advisor,
        description="Analyse une situation chantier et vérifie la conformité aux règles de sécurité.",
    )


regulatory_tool = create_regulatory_tool()

def analyze_camera(
    camera_name: str,
    images_folder: str,
    json_path: str,
) -> str:
    """
    Analyze images from a camera using detected objects and metadata to identify visible risks.
    If risks are detected, regulatory advice is fetched and images are annotated accordingly.

    Args:
        camera_name (str): Name of the camera.
        images_folder (str): Path to the folder containing images.
        json_path (str): Path to the JSON file with detection metadata.

    Returns:
        str: A combined summary report of the risk analysis for the camera.
    """

    # Load detection metadata from JSON
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Prepare the LLM and the prompt template (prompt in French)
    llm = ChatOllama(model="llama3:8b", temperature=0.2)
    prompt = ChatPromptTemplate.from_template(
        """Tu es un agent de securite.
        Images provenant de : {nom_camera}
        Voici les objets detectes :
        {objets}
        Indique s'il y a un risque visible. Sois bref et concis. Ne me dis pas bonjour.
        """
    )

    # Prepare output folder for annotated images
    annotated_folder = os.path.join(images_folder, "../annotated_images", camera_name)
    os.makedirs(annotated_folder, exist_ok=True)

    # Select images sorted alphabetically
    image_paths = [
        os.path.join(images_folder, filename)
        for filename in sorted(os.listdir(images_folder))
        if filename.lower().endswith(".jpg")
    ]

    def process_image(image_path: str) -> str:
        image_name = os.path.basename(image_path)
        info = metadata.get("images", {}).get(image_name, {})
        other_info = info.get("other")

        if other_info is None:
            return f"[{image_name}] Aucun risque détecté (aucune information 'other')."

        # Associate detected objects with risks if present
        objects = associate_risks(info.get("detections", []), other_info)

        # Format the detected objects for the prompt
        objects_text = (
            "\n".join(
                [
                    f"- {obj.get('label')} (score: {obj.get('score'):.2f})"
                    + (f" [Risque: {obj['risque']}]" if "risque" in obj else "")
                    for obj in objects
                ]
            )
            if objects
            else "Aucun objet detecte."
        )

        # Compose the prompt + LLM chain and get the response
        chain = prompt | llm
        response = chain.invoke({"nom_camera": camera_name, "objets": objects_text})
        text = response.content if hasattr(response, "content") else str(response)

        # If risk detected, get regulatory advice and annotate image
        regulatory_advice = regulatory_tool.func(
            f"Camera: {camera_name}\nImage: {image_name}\n{objects_text}\nAnalyse initiale: {text}"
        )
        annotate_image(image_path, objects, annotated_folder)

        print("REGULATORY ADVICE: ", regulatory_advice)
        return f"[ {image_name}]\n{text}\nVérification réglementaire : {regulatory_advice}\n"

    results = []
    # Process images concurrently with progress bar
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, path) for path in image_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"📷 {camera_name}"):
            results.append(future.result())

    # Return the combined summary report
    return f"\n=== Résumé {camera_name} ===\n" + "\n".join(results)