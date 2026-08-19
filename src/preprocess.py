import os
import cv2
import numpy as np
from pathlib import Path


VALID_PIC = {".jpg", ".jpeg", ".png"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def deskew_image(image, max_angle=30.0):
    """Deskew the image using OpenCV."""
    # Convert image to white text on black background and dilate each line to rectangular box for better contour detection
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 15))
    dilate = cv2.dilate(thresh, kernel)

    # find contours and obtain the set the angle of rotation as the median angle of the bounding boxes
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0.9 * image.shape[1] or w < 20 or h < 5:
            continue

        minAreaRect = cv2.minAreaRect(contour)
        angle = minAreaRect[-1]
        if angle != -90.0: 
            if angle < -45:
                angle = 90 + angle
            angles.append(angle)
    if not angles:
        return image
    angles.sort()

    mid_angle = angles[len(angles)//2]
    if abs(mid_angle) > max_angle:
        return image
    matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), mid_angle, 1)
    deskewed = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed


def normalize_brightness_contrast(image: np.ndarray) -> np.ndarray:
    """
    Normalize brightness/contrast to neutralize blue/yellow tint and lighting differences between photos taken at different times of day.
    """
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    normalized = clahe.apply(image)
    return normalized


def denoise(gray: np.ndarray) -> np.ndarray:
    """Light denoising reduces pencil smudges."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for better OCR results.
    Steps:
    1. Convert to grayscale
    2. Blur the image to increase stroke smoothness
    3. Deskew the image
    4. Normalize brightness and contrast
    5. Denoise the image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    deskewed = deskew_image(blurred)
    normalized = normalize_brightness_contrast(deskewed)
    processed = denoise(normalized)
    return processed

 
def iter_input_images(input_path: Path):
    """Return all valid image files in the input path."""
    if input_path.is_file():
        if input_path.suffix.lower() in VALID_PIC:
            yield input_path
        return
    for p in sorted(input_path.iterdir()):
        if p.suffix.lower() in VALID_PIC:
            yield p


def process_one_image(input_path: Path, output_dir: Path) -> Path:
    """Process one image and save the result to the output directory."""
    image = cv2.imread(str(input_path))
    processed = preprocess_image(image)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    cv2.imwrite(str(output_path), processed)
    return output_path


def main():
    input_path = Path(BASE_DIR) / '..' / 'data' / 'raw'
    output_dir = Path(BASE_DIR) / '..' / 'data' / 'processed'
    if not input_path.exists():
        print(f"Input path {input_path} does not exist.")
        return
    for image in iter_input_images(input_path):
        output_image = process_one_image(image, output_dir)


if __name__ == "__main__":
    main()