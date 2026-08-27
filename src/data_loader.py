import csv
import random
import torch
from PIL import Image, ImageOps
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / '..' / 'data'
pairs_csv_path = DATA_DIR / 'pairs' / 'all_pairs.csv'

WEIGHT_MAP = {
    'augmentation_positive': 0.7,
    'topic_positive': 1.0,
    'soft_positive': 0.5,
    'hard_negative': 0.0,
    'plain_negative': 0.0
}

negative_cap_multiplier = 1
positive_cap_multiplier = 1


def pad_to_square(image: Image.Image, size: int = 224) -> Image.Image:
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = image.resize((new_w, new_h), Image.LANCZOS)

    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return ImageOps.expand(img, border=(left, top, right, bottom), fill=255)


imageNet_mean = [0.485, 0.456, 0.406]
imageNet_std = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Lambda(lambda image: pad_to_square(image)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=imageNet_mean, std=imageNet_std)
])


class PairDataset(Dataset):
    def __init__(self, augmentation_only: bool, no_soft_positive: bool, pairs_csv: Path = pairs_csv_path, data_dir: Path = DATA_DIR, transform=image_transform):
        self.augmentation_only = augmentation_only
        self.no_soft_positive = no_soft_positive
        self.data_dir = Path(data_dir)
        self.transform = transform

        pairs_by_type = {}
        with open(pairs_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair_type = row['pair_type']
                pairs_by_type.setdefault(pair_type, []).append(row)

        self.plain_negatives = pairs_by_type.get('plain_negative')
        self.augmentation_positives = pairs_by_type.get('augmentation_positive')
        self.pairs = []

        if self.augmentation_only:
            self.include = self.augmentation_positives
        elif self.no_soft_positive:
            include_types = ('topic_positive', 'hard_negative')
            self.include = [row for pair_type, rows in pairs_by_type.items() if pair_type in include_types for row in rows]
        else:
            exclude_types = ('plain_negative', 'augmentation_positive')
            self.include = [row for pair_type, rows in pairs_by_type.items() if pair_type not in exclude_types for row in rows]


    def resample(self):
        base_count = len(self.include)
        cap_negative = base_count * negative_cap_multiplier
        cap_positive = base_count * positive_cap_multiplier

        if len(self.plain_negatives) > cap_negative:
            sampled_negative = random.sample(self.plain_negatives, cap_negative)
        else:
            sampled_negative = list(self.plain_negatives)

        if len(self.augmentation_positives) > cap_positive:
            sampled_positive = random.sample(self.augmentation_positives, cap_positive)
        else:
            sampled_positive = list(self.augmentation_positives)

        if self.augmentation_only:
            self.pairs = self.include + sampled_negative
        else:
            self.pairs = self.include + sampled_negative + sampled_positive

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row = self.pairs[index]
        img1 = Image.open(self.data_dir / row['img1'])
        img2 = Image.open(self.data_dir / row['img2'])

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        weight = WEIGHT_MAP[row['pair_type']]
        return img1, img2, torch.tensor(weight, dtype=torch.float32)



batch_size = 32
num_workers = 2
shuffle = True
persistent_workers = False

def build_dataloader(dataset: PairDataset) -> DataLoader:
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )