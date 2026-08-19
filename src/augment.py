"""
MathLens — augmentation pipeline.
 
Generates augmentation-positive variants of each cropped section (rotate, brightness/contrast jitter, crop) 
and records which files form a pair.
 
This pair type needs no manual labeling since every variant of a section is automatically a positive pair 
with that section and with every other variant of it.
"""
import os
import cv2
import csv
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path


VALID_PIC = {".jpg", ".jpeg", ".png"}
BASE_DIR = Path(__file__).parent


class RandomMorphology:
    """self.kernel_size must be odd."""
    def __init__(self, kernel_size: int = 3, p: float = 0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            arr = np.array(image)
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            if np.random.rand() < 0.5:
                arr = cv2.erode(arr, kernel, iterations=1)
            else:
                arr = cv2.dilate(arr, kernel, iterations=1)
            return Image.fromarray(arr)
        return image


class Augmentation:
    """Augmentation pipeline for images."""
    def __init__(
            self, 
            max_rotation: float = 5.0, 
            brightness: float = 0.3, 
            contrast: float = 0.3, 
            blur_prb: float = 0.5, 
            morphology_kernel: int = 3, 
            morphology_prb: float = 0.5
            ):
        self.morphology = RandomMorphology(kernel_size=morphology_kernel, p=morphology_prb)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=max_rotation, expand=True, fill=255),
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=blur_prb),
        ])

    def __call__(self, image: Image.Image) -> Image.Image:
        img = self.transform(image)
        img = self.morphology(img)
        return img


def find_section_file(section_dir: Path, section_id: str) -> Path:
    """Find the image path for a given section ID in the specified directory."""
    for format in VALID_PIC:
        file_path = section_dir / f"{section_id}{format}"
        if file_path.exists():
            return file_path
    raise FileNotFoundError(f"No valid image file found for section ID {section_id} in {section_dir}")


def read_section_ids(metadata_path: Path) -> list:
    """Read section IDs from a metadata file."""
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row['section_id'] for row in reader if row.get('section_id')]


def generate_variants_for_one(section_id:str, section_dir: Path, output_dir: Path, augmenter: Augmentation, n_variants: int) -> list:
    """Generate augmented variants for a given section ID and save them to the output directory."""
    section_path = find_section_file(section_dir, section_id)
    image = Image.open(section_path)
    out_subdir = output_dir / section_id
    out_subdir.mkdir(parents=True, exist_ok=True)

    variant_paths = []
    for i in range(n_variants):
        augmented_image = augmenter(image)
        variant_id = f"{section_id}_aug{i+1}"
        variant_path = out_subdir / f"{variant_id}.jpg"
        augmented_image.save(variant_path)
        variant_paths.append((variant_id, variant_path))
    return variant_paths


def main():
    section_dir = BASE_DIR / '..' / 'data' / 'sections'
    output_dir = BASE_DIR / '..' / 'data' / 'augmented'
    pairs_out = BASE_DIR / '..' / 'data' / 'pairs' / 'augmented_pairs.csv'
    metadata_path = BASE_DIR / '..' / 'data' / 'pairs' / 'metadata.csv'

    n_variants = 4
    augmenter = Augmentation(max_rotation=5.0, brightness=0.3, contrast=0.3, blur_prb=0.5, morphology_kernel=3, morphology_prb=0.5)
    section_ids = read_section_ids(metadata_path)

    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['section_id', 'variant_id', 'variant_path'])

        for section_id in section_ids:
            try:
                variant_paths = generate_variants_for_one(section_id, section_dir, output_dir, augmenter, n_variants)
            except FileNotFoundError as e:
                print(f'[skip] {e}')
                continue
            for variant_id, variant_path in variant_paths:
                writer.writerow([section_id, variant_id, str(variant_path)])


if __name__ == "__main__":
    main()