import cv2
import os
import re
import numpy as np
import json
from pdf2image import convert_from_path
import difflib
from paddleocr import PaddleOCR

# Initialisation de l'OCR Paddle avec reconnaissance d'angle (pour corriger les textes inclinés),
# détection activée (det=True), et l’algorithme CRNN pour une reconnaissance robuste.
ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec_algorithm='CRNN')

# Configuration manuelle du chemin vers les binaires Poppler (nécessaire à pdf2image sous Windows)
os.environ["PATH"] += os.pathsep + r"H:\\Desktop\\projet_semestre\\NotreProjet\\poppler\\poppler-24.08.0\\Library\\bin"

def normalize(text):
    """
    Supprime tous les caractères non alphanumériques et met le texte en majuscules.
    Permet de comparer plus facilement des titres OCR approximatifs à des titres de référence.
    """
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def postprocess_titles(flat_titles):
    """
    Corrige et structure les titres extraits par OCR avant de les intégrer comme métadonnées du tableau.
    
    Étapes :
    - Remplacement des erreurs typographiques communes (ex: IOOC -> 1000)
    - Fusion des morceaux de titres répartis sur plusieurs lignes
    - Séparation des années concaténées (ex: "19021903")
    - Fusion des expressions "1900 to 1904." qui peuvent apparaître fragmentées
    """

    # Correction d’erreurs d’OCR typiques dans les titres
    flat_titles = [t.replace("IOOC", "1000") for t in flat_titles]

    # Étape 1 : Fusion de fragments de titres reconnus en un seul
    corrected = []
    i = 0
    while i < len(flat_titles):
        text = flat_titles[i]
        if i + 1 < len(flat_titles):
            combined = text + " " + flat_titles[i+1]
            combined_norm = normalize(combined)

            for full_title in [
                "NUMBER OF DEATHS FROM ALL CAUSES PER 1000 OF POPULATION.",
                "Annual average: 1900 to 1904."
            ]:
                if normalize(full_title) in combined_norm:
                    corrected.append(full_title)
                    i += 2
                    break
            else:
                corrected.append(text)
                i += 1
        else:
            corrected.append(text)
            i += 1

    # Étape 2 : Séparation des chaînes d'années concaténées comme "19021903"
    split_corrected = []
    for t in corrected:
        if re.fullmatch(r'\d{8}', t):
            split_corrected.extend([t[:4], t[4:]])
        else:
            split_corrected.append(t)

    # Étape 3 : Fusion des paires "1900to", "1904." → "1900 to 1904."
    final_titles = []
    i = 0
    while i < len(split_corrected):
        if (i + 1 < len(split_corrected) and
            re.fullmatch(r'\d{4}to', split_corrected[i].replace(" ", "")) and
            re.fullmatch(r'\d{4}\.', split_corrected[i+1])):
            year1 = re.sub(r'[^0-9]', '', split_corrected[i])
            year2 = re.sub(r'[^0-9]', '', split_corrected[i+1])
            final_titles.append(f"{year1} to {year2}.")
            i += 2
        else:
            final_titles.append(split_corrected[i])
            i += 1

    return final_titles

 
 def apply_title_matching(rows):
    """
    Prend les lignes OCR groupées et identifie la portion correspondant aux titres.
    Le reste est traité comme données tabulaires.
    
    Règle : les lignes avant la première ligne contenant au moins une valeur numérique sont considérées comme titre.
    """
    title_rows = []
    data_start = 0
    for i, row in enumerate(rows):
        # Détection de la première ligne contenant une ou plusieurs valeurs numériques
        if any(re.fullmatch(r"\d{1,3}(\.\d+)?", cell) for cell in row):
            data_start = i
            break
        title_rows.append(row)

    # Fusion et nettoyage des fragments de titres
    raw_titles = [cell for row in title_rows for cell in row]
    processed_titles = postprocess_titles(raw_titles)

    # On retourne la concaténation des titres corrigés + les données du tableau
    return processed_titles + rows[data_start:]

def extract_text_raw(roi):
    # Applique PaddleOCR sur la région d'intérêt (ROI) du tableau
    # Chaque ligne détectée par l'OCR est analysée pour extraire :
    # - le texte reconnu
    # - la position du texte (coordonnées centrales pour regroupement spatial)

    results = ocr.ocr(roi, cls=True)  # cls=True active la classification d'orientation du texte
    raw_data = []

    for line in results:
        for box, (text, _) in line:
            (x1, y1), (x2, y2), _, _ = box  # on récupère les coordonnées du rectangle englobant le texte
            cx = int((x1 + x2) / 2)  # calcul de la position horizontale centrale
            cy = int((y1 + y2) / 2)  # calcul de la position verticale centrale
            raw_data.append({'x': cx, 'y': cy, 'text': text})  # sauvegarde de chaque cellule OCR avec sa position

    return raw_data

def group_text_into_rows(raw_data, y_threshold=10):
    # Cette fonction regroupe les textes OCR en lignes en fonction de leur proximité verticale.
    # Elle suppose que les textes appartenant à la même ligne ont des 'y' proches.

    raw_data.sort(key=lambda x: x['y'])  # trie les éléments par position verticale
    rows, current_row, current_y = [], [], None

    for item in raw_data:
        if current_y is None:
            current_y = item['y']
        if abs(item['y'] - current_y) <= y_threshold:
            # Même ligne : ajout de l'élément
            current_row.append(item)
        else:
            # Nouvelle ligne : enregistre l'ancienne, initialise une nouvelle
            rows.append(current_row)
            current_row = [item]
            current_y = item['y']
    if current_row:
        rows.append(current_row)

    # Trie les éléments dans chaque ligne horizontalement
    for row in rows:
        row.sort(key=lambda x: x['x'])

    # Formate les lignes : seulement les textes
    grouped_rows = [[cell['text'] for cell in row] for row in rows]

    print("--- OCR Rows ---")
    for row in grouped_rows:
        print(row)

    # Applique un traitement des titres (ex: fusion, normalisation, corrections)
    return apply_title_matching(grouped_rows)

def clean_text(cell):
    # Nettoie une cellule textuelle :
    # - supprime les espaces
    # - remplace les virgules par des points (standardise les décimales)
    if not isinstance(cell, str):
        return cell
    return cell.replace(" ", "").replace(",", ".")

def is_year(value):
    # Détermine si une cellule correspond à une année (format : 19XX ou 20XX)
    return isinstance(value, str) and re.fullmatch(r'19\d{2}|20\d{2}', value)

def is_numeric_value(val):
    # Cette fonction étend la détection des valeurs numériques à :
    # - float (type réel en Python)
    # - chaînes représentant un float (ex: "18.4")
    # - chaînes représentant des formats tronqués (ex: "18.")
    # - None (pour gérer les cellules vides ou manquantes)
    
    if val is None:
        return True
    if isinstance(val, float):
        return True
    if isinstance(val, str):
        return bool(re.fullmatch(r"\d{1,3}\.\d+", val) or re.fullmatch(r"\d{1,3}\.", val))
    return False

def clean_and_format_cell(cell):
    # Cette fonction nettoie et transforme une cellule brute OCR :
    # - élimine les cas de cellules invalides ou bruitées
    # - convertit des chaînes comme "182" en float 18.2 (présumé)
    # - garde les années intactes
    # - garde les autres textes inchangés

    if not cell or cell in {"1", "(1", "(!)", "(1)", "("}:
        return None  # valeurs ignorées souvent issues de notes de bas de page

    cell = clean_text(cell)

    # Cas standard float explicite : "12.3" => 12.3
    if re.fullmatch(r'\d{1,3}\.\d+', cell):
        return float(cell)

    # Cas de valeurs compactées : "182" => 18.2 (inférence)
    if re.fullmatch(r'\d{2,3}', cell):
        return float(cell[:-1] + '.' + cell[-1])

    # Cas des années : laissé tel quel
    if re.fullmatch(r'\d{4}', cell):
        return cell

    # Sinon, renvoyer tel quel (texte)
    return cell

def detect_column_types(rows):
    # Objectif : déterminer dynamiquement le nombre de colonnes textuelles et numériques
    # à partir des données du tableau nettoyées.
    
    def is_valid_text(c):
        # Une cellule est considérée comme texte si :
        # - c'est une string
        # - ce n'est pas une année
        # - elle est suffisamment longue (plus de 3 caractères)
        # - ce n'est pas un float déguisé avec un point final (ex: "18.")
        return isinstance(c, str) and not is_year(c) and len(c) > 3 and not re.match(r'^\d{1,3}\.$', c)

    # On cherche la ligne avec le plus de textes valides pour estimer les colonnes texte
    max_row_text = max(
        (r for r in rows if any(is_valid_text(c) for c in r)),
        key=lambda r: sum(is_valid_text(c) for c in r),
        default=[]
    )

    # Idem pour les colonnes numériques
    max_row_num = max(
        (r for r in rows if any(is_numeric_value(c) for c in r)),
        key=lambda r: sum(is_numeric_value(c) for c in r),
        default=[]
    )

    text_cols = sum(is_valid_text(c) for c in max_row_text)
    num_cols = sum(is_numeric_value(c) for c in max_row_num)

    print(f"Nombre de colonnes num: {num_cols}")
    print(f"Nombre de colonnes text: {text_cols}")

    return text_cols, num_cols

def is_title(row):
    # Une ligne est considérée comme un titre si elle ne contient **aucun float**
    return all(not isinstance(c, float) for c in row)

def is_float(val):
    # Fonction utilitaire pour reconnaître un float Python ou une chaîne représentant un float
    try:
        return (
            isinstance(val, float)
            or (isinstance(val, str) and re.fullmatch(r"\d{1,3}\.\d+", val))
        )
    except:
        return False

def split_titles_and_data(rows):
    # Cette fonction sépare le tableau en deux parties :
    # - les lignes de titre (avant les données)
    # - les lignes de données (dès que des valeurs numériques sont détectées)
    titles = []
    data_start_index = 0

    for i, row in enumerate(rows):
        floats = sum(1 for cell in row if is_float(cell))
        texts = sum(1 for cell in row if isinstance(cell, str) and not is_year(cell))

        # Heuristique : une ligne contenant au moins deux floats + un texte
        # est considérée comme une ligne de données
        if floats >= 2 and texts >= 1:
            data_start_index = i
            break
        titles.append(row)

    return titles, rows[data_start_index:]


def process_and_format_rows(rows):
    # Fonction centrale de structuration des lignes de données

    # 1. Nettoyage de chaque cellule
    cleaned = [[clean_and_format_cell(cell) for cell in row] for row in rows]

    # 2. Détection des types de colonnes
    text_cols, num_cols = detect_column_types(cleaned)

    result = []
    i = 0

    # 3. Fusion des lignes textuelles incomplètes avec la ligne suivante
    while i < len(cleaned):
        row = cleaned[i]

        if is_title(row):
            result.append(row)
            i += 1
            continue

        text_values = [c for c in row if isinstance(c, str) and not is_year(str(c))]
        numeric_count = sum(isinstance(c, float) for c in row)

        if len(text_values) == 1 and numeric_count == 0:
            # Cas : ligne contenant uniquement un libellé, sans données
            # -> fusionner avec la suivante
            if i + 1 < len(cleaned):
                next_row = cleaned[i + 1]
                next_texts = [c for c in next_row if isinstance(c, str) and not is_year(str(c))]
                if len(next_texts) < text_cols:
                    cleaned[i + 1] = text_values + next_row
                    i += 2
                    continue

        elif len(text_values) < text_cols and result:
            # Si la ligne actuelle est incomplète (libellé partiel),
            # on complète avec les valeurs de la ligne précédente
            last_texts = [c for c in result[-1] if isinstance(c, str) and not is_year(str(c))]
            if len(last_texts) >= (text_cols - len(text_values)):
                row = last_texts[:text_cols - len(text_values)] + row

        result.append(row)
        i += 1

    # 4. Fusion des lignes numériques incomplètes
    i = 0
    while i < len(result) - 1:
        current_row = result[i]
        next_row = result[i + 1]

        current_num_count = sum(isinstance(cell, float) for cell in current_row)
        next_num_count = sum(isinstance(cell, float) for cell in next_row)

        # Fusion possible si la ligne suivante contient uniquement des floats
        # et que la somme des colonnes est toujours dans les limites
        if current_num_count + next_num_count <= num_cols:
            merged_row = current_row + [cell for cell in next_row if isinstance(cell, float)]
            result[i] = merged_row
            del result[i + 1]
        else:
            i += 1

    # 5. Ajout de valeurs nulles si une ligne a trop peu de colonnes numériques
    for row in result:
        numeric_count = sum(1 for cell in row if is_numeric_value(cell))
        if numeric_count < num_cols:
            row.extend([None] * (num_cols - numeric_count))

    return result

def extract_table_data(image, table_contours, output_folder):
    # Objectif : extraire et structurer les données OCR pour chaque tableau détecté sur une page d'image.
    
    structured_data = {}  # Dictionnaire de sortie contenant tous les tableaux d'une image

    for i, (x, y, w, h) in enumerate(table_contours):
        # Pour chaque contour identifié comme un tableau, on extrait la ROI (Region of Interest)
        roi = image[y:y+h, x:x+w]

        # Etape 1 : OCR brut sur la région du tableau
        raw_ocr = extract_text_raw(roi)

        # Etape 2 : Regroupement vertical/horizontal des cellules OCR pour former des lignes
        table_data = group_text_into_rows(raw_ocr)

        # Etape 3 : Séparation entre titres et données
        titles, data_rows = split_titles_and_data(table_data)

        # Etape 4 : Nettoyage, normalisation, fusion et formatage final
        final_data = process_and_format_rows(data_rows)

        # On assemble les titres et les lignes dans la structure de sortie
        structured_data[f"Tableau {i+1}"] = [*titles, *final_data]

    return structured_data

def process_pdf(pdf_path, output_folder, output_json):
    # Objectif : pipeline complet de traitement d'un fichier PDF
    # Entrée : chemin vers le PDF
    # Sortie : dictionnaire structuré de tous les tableaux détectés

    # Conversion des pages du PDF en images avec une haute résolution (300 dpi)
    images = convert_from_path(pdf_path, dpi=300)

    structured_tables = {}  # Dictionnaire final pour toutes les pages

    for i, img in enumerate(images):
        # Conversion PIL -> OpenCV (BGR)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Détection des zones de tableau sur la page courante
        tables = detect_tables(img_cv)

        # Extraction et structuration des données OCR dans chaque tableau détecté
        extracted_data = extract_table_data(img_cv, tables, output_folder)

        # Sauvegarde des données extraites pour cette page
        structured_tables[f"Page {i+1}"] = extracted_data

    # Sauvegarde finale du dictionnaire complet dans un fichier JSON (lisible humainement)
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(structured_tables, json_file, ensure_ascii=False, indent=4)

    return structured_tables

