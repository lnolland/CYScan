import cv2
import os
import re
import numpy as np
import json
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# Initialisation de PaddleOCR optimisé pour les chiffres et le texte
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det=True,
    rec_algorithm='CRNN'
)

# Configuration de Poppler
os.environ["PATH"] += os.pathsep + r"H:\\Desktop\\projet_semestre\\NotreProjet\\poppler\\poppler-24.08.0\\Library\\bin"

def merge_contours(contours, threshold=50):
    """Fusionne les contours qui se chevauchent ou sont proches."""
    if not contours:
        return []
    
    merged = []
    for cnt in contours:
        x, y, w, h = cnt
        merged_flag = False
        
        for i, (mx, my, mw, mh) in enumerate(merged):
            if (x < mx + mw + threshold and x + w > mx - threshold and
                y < my + mh + threshold and y + h > my - threshold):
                new_x = min(x, mx)
                new_y = min(y, my)
                new_w = max(x + w, mx + mw) - new_x
                new_h = max(y + h, my + mh) - new_y
                merged[i] = (new_x, new_y, new_w, new_h)
                merged_flag = True
                break
        
        if not merged_flag:
            merged.append(cnt)
    
    return merged

def detect_tables(image):
    """Détecte les tables et applique une légère expansion pour éviter les coupures."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(img_gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Détection des lignes horizontales et verticales
    kernel_h = np.ones((1, 50), np.uint8)
    kernel_v = np.ones((50, 1), np.uint8)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

    # Fusion des lignes détectées
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    morph = cv2.dilate(table_mask, np.ones((3, 3), np.uint8), iterations=2)

    # Détection des contours des tableaux
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = [(cv2.boundingRect(contour)) for contour in contours if cv2.boundingRect(contour)[2] > 200 and cv2.boundingRect(contour)[3] > 100]

    # Fusion des fragments de tableaux
    merged_contours = merge_contours(detected_contours, threshold=30)

    # Légère expansion des tableaux détectés
    expanded_contours = []
    padding = 10  # Expansion légère pour éviter la coupure des bords
    for (x, y, w, h) in merged_contours:
        new_x = max(x - padding, 0)
        new_y = max(y - padding, 0)
        new_w = min(w + 2 * padding, image.shape[1] - new_x)  # Vérifier pour ne pas dépasser les bords
        new_h = min(h + 2 * padding, image.shape[0] - new_y)
        expanded_contours.append((new_x, new_y, new_w, new_h))

    return expanded_contours

def extract_text_from_roi(roi):
    """Utilise PaddleOCR pour extraire le texte et les nombres depuis une région spécifique."""
    results = ocr.ocr(roi, cls=True)
    extracted_text = []
    for line in results:
        row = [entry[1][0] for entry in line]
        extracted_text.append(row)
    return extracted_text

def extract_table_data(image, table_contours, output_folder):
    """Extrait les tableaux et applique PaddleOCR."""
    structured_data = {}
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (x, y, w, h) in enumerate(table_contours):
        roi = image[y:y+h, x:x+w]

        # Sauvegarde de l'image extraite
        output_path = os.path.join(output_folder, f"extracted_table_{i+1}.png")
        cv2.imwrite(output_path, roi)
        print(f"Image extraite et sauvegardée : {output_path}")

        # Affichage pour validation
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Tableau {i+1} extrait")
        plt.show()

        # Application de PaddleOCR sur l'image extraite
        table_data = extract_text_from_roi(roi)

        # Stockage des résultats
        structured_data[f"Tableau {i+1}"] = table_data

    return structured_data

def clean_and_format_cell(cell):
    """
    Nettoie et formate une cellule extraite par OCR.
    - Convertit les nombres mal formatés.
    - Identifie les textes.
    - Gère les valeurs manquantes.
    """
    if not cell or cell in {"1", "(1", "(!)"}:  # Valeurs pouvant représenter un manque de données
        return None
    
    # Supprimer les espaces à l'intérieur des nombres mal séparés
    cell = cell.replace(" ", "")
    
    # Vérifier si c'est un nombre
    if re.match(r'^\d{1,3}\.\d$', cell) or re.match(r'^\d{2,3}\.\d{1,3}$', cell):
        return float(cell)  # Déjà bien formaté
    
    # Vérifier les nombres mal interprétés
    if re.match(r'^\d{2,3}$', cell):  # XX ou XXX (ex: "23" ou "234" doivent être "2.3" ou "23.4")
        return float(cell[:-1] + '.' + cell[-1])
    
    if re.match(r'^\d{4}$', cell):  # XXXX (ex: "2024" doit rester un texte car c'est une année)
        return cell  # On garde comme texte
    
    return cell  # Retourner tel quel s'il ne correspond pas aux critères

def format_table_data(table_data):
    """Applique le formatage à chaque cellule du tableau OCR extrait."""
    formatted_table = []
    for row in table_data:
        formatted_row = [clean_and_format_cell(cell) for cell in row]
        formatted_table.append(formatted_row)
    return formatted_table

def process_pdf(pdf_path, output_folder, output_json):
    """Convertit le PDF en images, détecte les tables et applique OCR."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path, dpi=300)  # DPI ajusté pour plus de netteté
    structured_tables = {}
    
    for i, img in enumerate(images):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        tables = detect_tables(img_cv)
        extracted_data = extract_table_data(img_cv, tables, output_folder)

        # Appliquer le formatage des données après l'extraction
        formatted_data = {
            table_name: format_table_data(content)
            for table_name, content in extracted_data.items()
        }
        
        structured_tables[f"Page {i+1}"] = formatted_data

    # Sauvegarde du JSON OCR
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(structured_tables, json_file, ensure_ascii=False, indent=4)

    return structured_tables

# Exécution
pdf_path = "test.pdf"
output_folder = "output_images"
output_json = "structured_extracted_data.json"
data_extracted = process_pdf(pdf_path, output_folder, output_json)

# Affichage des résultats OCR
for page, tables in data_extracted.items():
    print(f"\n{page}")
    for table_name, content in tables.items():
        print(f"\n{table_name} :")
        for row in content:
            print(row)
