import cv2
import numpy as np
import os
from collections import defaultdict


def extract_handwritten_digits(image_path, output_base_folder="extracted_digits"):
    """
    Extract handwritten digits from an image and save them in organized folders

    Args:
        image_path: Path to the input image
        output_base_folder: Base folder where digit folders will be created
    """

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            if 0.2 < aspect_ratio < 2.0 and w > 10 and h > 10:
                digit_contours.append(contour)

    def get_contour_position(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return (y // 50, x)

    digit_contours.sort(key=get_contour_position)

    for digit in range(10):
        digit_folder = os.path.join(output_base_folder, str(digit))
        os.makedirs(digit_folder, exist_ok=True)

    n_cols = 10

    cx_items = []
    for c in digit_contours:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2.0
        cx_items.append((cx, c))
    cx_items.sort(key=lambda t: t[0])

    gaps = []
    for i in range(len(cx_items) - 1):
        gaps.append((cx_items[i + 1][0] - cx_items[i][0], i))
    gaps.sort(reverse=True)
    split_indices = sorted([i for _, i in gaps[: max(0, n_cols - 1)]])
    split_set = set(split_indices)

    contour_to_col = {}
    col = 0
    for i, (_, c) in enumerate(cx_items):
        if i > 0 and (i - 1) in split_set:
            col += 1
        contour_to_col[id(c)] = min(col, n_cols - 1)

    rows = defaultdict(list)
    for contour in digit_contours:
        x, y, w, h = cv2.boundingRect(contour)
        row = y // 50
        rows[row].append((contour, x, y, w, h))

    for row in rows.values():
        row.sort(key=lambda item: item[1])

    digit_counts = [0] * 10

    for row_idx, row_data in enumerate(sorted(rows.items())):
        row_num, contours_in_row = row_data

        for _, (contour, x, y, w, h) in enumerate(contours_in_row):

            digit_roi = thresh[y : y + h, x : x + w]

            max_dim = max(w, h)
            square_roi = np.zeros((max_dim, max_dim), dtype=np.uint8)

            start_x = (max_dim - w) // 2
            start_y = (max_dim - h) // 2
            square_roi[start_y : start_y + h, start_x : start_x + w] = digit_roi

            resized_digit = cv2.resize(
                square_roi, (28, 28), interpolation=cv2.INTER_AREA
            )

            digit_label = contour_to_col.get(id(contour))
            if digit_label is None:
                continue

            filename = f"digit_{digit_counts[digit_label]:03d}.jpg"
            output_path = os.path.join(output_base_folder, str(digit_label), filename)

            cv2.imwrite(output_path, resized_digit)
            digit_counts[digit_label] += 1

            print(f"Saved digit {digit_label} as {output_path}")

    print(f"\nExtraction complete!")
    print(f"Total digits extracted by category:")
    for i in range(10):
        print(f"Digit {i}: {digit_counts[i]} images")


def visualize_extraction_process(image_path):
    """
    Visualize the digit detection process for debugging
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    visualization = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 2.0 and w > 10 and h > 10:
                cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("digit_detection_visualization.jpg", visualization)
    print("Saved digit detection visualization as 'digit_detection_visualization.jpg'")


if __name__ == "__main__":

    image_path = "digit_sheet.jpg"

    print("Creating visualization of digit detection...")
    visualize_extraction_process(image_path)

    print("\nExtracting digits...")
    extract_handwritten_digits(image_path)

    print("\nDone! Check the 'extracted_digits' folder for results.")