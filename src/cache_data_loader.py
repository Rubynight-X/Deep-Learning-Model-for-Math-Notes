import csv
import torch
import random
from pathlib import Path
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from data_loader import image_transform
from torch.utils.data import Dataset, DataLoader


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / '..' / 'data'
PAIRS_CSV = DATA_DIR / 'pairs' / 'all_pairs.csv'
CACHE_PATH = DATA_DIR / 'features_cache.pt'


def cache_features():
    paths = set()
    with open (PAIRS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.add(row['img1'])
            paths.add(row['img2'])
    print(f'Unique images to cache: {len(paths)}')


    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    print(f'Device: {device}')

    cache = {}
    with torch.no_grad():
        for i, path in enumerate(sorted(paths)):
            img = Image.open(DATA_DIR / path)
            tensor = image_transform(img).unsqueeze(0).to(device)
            features = backbone(tensor).squeeze(0).cpu()
            cache[path] = features

    torch.save(cache, CACHE_PATH)
    print(f'Saved cache to {CACHE_PATH}')
    print(f'Each vector: {list(cache.values())[0].shape}')



WEIGHT_MAP = {
    'augmentation_positive': 0.7,
    'topic_positive': 1.0,
    'soft_positive': 0.5,
    'hard_negative': 0.0,
    'plain_negative': 0.0
}

negative_cap_multiplier = 5
positive_cap_multiplier = 3


class CachedPairDataset(Dataset):
    def __init__(self, augmentation_only: bool, no_soft_positive: bool, pairs_csv: Path = PAIRS_CSV, cache_path: Path = CACHE_PATH):
        self.augmentation_only = augmentation_only
        self.no_soft_positive = no_soft_positive

        self.cache = torch.load(cache_path, weights_only=True)

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
        feature1 = self.cache[row['img1']]
        feature2 = self.cache[row['img2']]
        weight = WEIGHT_MAP[row['pair_type']]
        return feature1, feature2, torch.tensor(weight, dtype=torch.float32)


batch_size = 32
num_workers = 0
shuffle = True
persistent_workers = False

def build_dataloader(dataset: CachedPairDataset) -> DataLoader:
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )


if __name__ == '__main__':
    cache_features()
