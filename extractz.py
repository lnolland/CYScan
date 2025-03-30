import cv2
import os
import re
import numpy as np
import json
from pdf2image import convert_from_path
import difflib
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec_algorithm='CRNN')
os.environ["PATH"] += os.pathsep + r"H:\\Desktop\\projet_semestre\\NotreProjet\\poppler\\poppler-24.08.0\\Library\\bin"

def normalize(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def find_all_title_matches(concat_text, pool):
    matches = []
    normalized_text = normalize(concat_text)
    for ref in pool:
        ref_text = normalize(' '.join(ref))
        if ref_text in normalized_text:
            matches.append(ref)
    return matches

def postprocess_titles(flat_titles):
    flat_titles = [t.replace("IOOC", "1000") for t in flat_titles]

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

    split_corrected = []
    for t in corrected:
        if re.fullmatch(r'\d{8}', t):
            split_corrected.extend([t[:4], t[4:]])
        else:
            split_corrected.append(t)

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
    title_rows = []
    data_start = 0
    for i, row in enumerate(rows):
        if any(re.fullmatch(r"\d{1,3}(\.\d+)?", cell) for cell in row):
            data_start = i
            break
        title_rows.append(row)

    raw_titles = [cell for row in title_rows for cell in row]
    processed_titles = postprocess_titles(raw_titles)
    return processed_titles + rows[data_start:]


def detect_tables(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(img_gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_h = np.ones((1, 50), np.uint8)
    kernel_v = np.ones((50, 1), np.uint8)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    morph = cv2.dilate(table_mask, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = [(cv2.boundingRect(contour)) for contour in contours if cv2.boundingRect(contour)[2] > 200 and cv2.boundingRect(contour)[3] > 100]
    return merge_contours(detected_contours, threshold=30)

def merge_contours(contours, threshold=50):
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

def extract_text_raw(roi):
    results = ocr.ocr(roi, cls=True)
    raw_data = []
    for line in results:
        for box, (text, _) in line:
            (x1, y1), (x2, y2), _, _ = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            raw_data.append({'x': cx, 'y': cy, 'text': text})
    return raw_data

def group_text_into_rows(raw_data, y_threshold=10):
    raw_data.sort(key=lambda x: x['y'])
    rows, current_row, current_y = [], [], None
    for item in raw_data:
        if current_y is None:
            current_y = item['y']
        if abs(item['y'] - current_y) <= y_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
            current_y = item['y']
    if current_row:
        rows.append(current_row)
    for row in rows:
        row.sort(key=lambda x: x['x'])
    grouped_rows = [[cell['text'] for cell in row] for row in rows]
    print("--- OCR Rows ---")
    for row in grouped_rows:
        print(row)
    return apply_title_matching(grouped_rows)

def clean_text(cell):
    if not isinstance(cell, str):
        return cell
    return cell.replace(" ", "").replace(",", ".")

def is_year(value):
    return isinstance(value, str) and re.fullmatch(r'19\d{2}|20\d{2}', value)

def is_numeric_value(val):
    if val is None:
        return True
    if isinstance(val, float):
        return True
    if isinstance(val, str):
        return bool(re.fullmatch(r"\d{1,3}\.\d+", val) or re.fullmatch(r"\d{1,3}\.", val))
    return False


def clean_and_format_cell(cell):
    if not cell or cell in {"1", "(1", "(!)", "(1)", "("}:
        return None
    cell = clean_text(cell)
    if re.fullmatch(r'\d{1,3}\.\d+', cell):
        return float(cell)
    if re.fullmatch(r'\d{2,3}', cell):
        return float(cell[:-1] + '.' + cell[-1])
    if re.fullmatch(r'\d{4}', cell):
        return cell
    return cell

def detect_column_types(rows):
    def is_valid_text(c):
        return isinstance(c, str) and not is_year(c) and len(c) > 3 and not re.match(r'^\d{1,3}\.$', c)
    max_row_text = max((r for r in rows if any(is_valid_text(c) for c in r)), key=lambda r: sum(is_valid_text(c) for c in r), default=[])
    max_row_num = max((r for r in rows if any(isinstance(c, float) for c in r)), key=lambda r: sum(isinstance(c, float) for c in r), default=[])
    text_cols = sum(is_valid_text(c) for c in max_row_text)
    num_cols = sum(isinstance(c, float) for c in max_row_num)
    print(f"Nombre de colonnes num: {num_cols}")
    print(f"Nombre de colonnes text: {text_cols}")
    return text_cols, num_cols

def is_title(row):
    return all(not isinstance(c, float) for c in row)

def is_float(val):
    try:
        return isinstance(val, float) or (isinstance(val, str) and re.fullmatch(r"\d{1,3}\.\d+", val))
    except:
        return False

def split_titles_and_data(rows):
    titles = []
    data_start_index = 0
    for i, row in enumerate(rows):
        floats = sum(1 for cell in row if is_float(cell))
        texts = sum(1 for cell in row if isinstance(cell, str) and not is_year(cell))
        if floats >= 2 and texts >= 1:
            data_start_index = i
            break
        titles.append(row)
    return titles, rows[data_start_index:]

def process_and_format_rows(rows):
    cleaned = [[clean_and_format_cell(cell) for cell in row] for row in rows]
    text_cols, num_cols = detect_column_types(cleaned)
    result = []
    i = 0
    while i < len(cleaned):
        row = cleaned[i]
        if is_title(row):
            result.append(row)
            i += 1
            continue
        text_values = [c for c in row if isinstance(c, str) and not is_year(str(c))]
        if len(text_values) == 1 and sum(isinstance(c, float) for c in row) == 0:
            if i + 1 < len(cleaned):
                next_row = cleaned[i + 1]
                next_texts = [c for c in next_row if isinstance(c, str) and not is_year(str(c))]
                if len(next_texts) < text_cols:
                    cleaned[i + 1] = text_values + next_row
                    i += 2
                    continue
        elif len(text_values) < text_cols and result:
            last_texts = [c for c in result[-1] if isinstance(c, str) and not is_year(str(c))]
            if len(last_texts) >= (text_cols - len(text_values)):
                row = last_texts[:text_cols - len(text_values)] + row
        result.append(row)
        i += 1

    i = 0
    while i < len(result) - 1:
        current_row = result[i]
        next_row = result[i + 1]
        current_num_count = sum(isinstance(cell, float) for cell in current_row)
        next_num_count = sum(isinstance(cell, float) for cell in next_row)
        if current_num_count + next_num_count <= num_cols:
            merged_row = current_row + [cell for cell in next_row if isinstance(cell, float)]
            result[i] = merged_row
            del result[i + 1]
        else:
            i += 1
    
    for row in result:
        numeric_count = sum(1 for cell in row if is_numeric_value(cell))
        if numeric_count < num_cols:
            row.extend([None] * (num_cols - numeric_count))
            
    return result

def extract_table_data(image, table_contours, output_folder):
    structured_data = {}
    for i, (x, y, w, h) in enumerate(table_contours):
        roi = image[y:y+h, x:x+w]
        raw_ocr = extract_text_raw(roi)
        table_data = group_text_into_rows(raw_ocr)
        titles, data_rows = split_titles_and_data(table_data)
        final_data = process_and_format_rows(data_rows)
        structured_data[f"Tableau {i+1}"] = [*titles, *final_data]
    return structured_data

def process_pdf(pdf_path, output_folder, output_json):
    images = convert_from_path(pdf_path, dpi=300)
    structured_tables = {}
    for i, img in enumerate(images):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        tables = detect_tables(img_cv)
        extracted_data = extract_table_data(img_cv, tables, output_folder)
        structured_tables[f"Page {i+1}"] = extracted_data
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(structured_tables, json_file, ensure_ascii=False, indent=4)
    return structured_tables
