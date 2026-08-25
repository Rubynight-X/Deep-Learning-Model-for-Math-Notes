import csv
import torch
from pathlib import Path
from augment import Augmentation
from cache_train import EmbeddingHead
from data_loader import image_transform
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


BASE_DIR = Path(__file__).parent / '..'
DATA_DIR = BASE_DIR / 'data'
CACHE_PATH = DATA_DIR / 'features_cache.pt'
METAPATA_PATH = DATA_DIR / 'pairs' / 'metadata.csv'
NUM_VARIANT = 4

EXPERIMENT = 'A'

def load_metadata():
    metadata = {}
    with open(METAPATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['section_id']] = row
    return metadata


def evaluate_augmentation(experiment):
    checkpoint_path = BASE_DIR / 'cached' / 'checkpoints_cached' / f'experiment_{experiment}_reduced_cap' / 'best.pt'
    embeddings_path = DATA_DIR / 'embeddings' / f'embeddings_{experiment}_reduced_cap.pt'
    print(f'Evaluating augmentation invariance - Experiment {experiment}')
    print(f'Checkpoint: {checkpoint_path}')
    print()

    metadata = load_metadata()
    head = EmbeddingHead()
    head.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location='cpu'))
    head.eval()

    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval()

    embeddings = torch.load(embeddings_path, weights_only=True, map_location='cpu')
    all_section_ids = list(embeddings.keys())
    all_section_embs = list(embeddings.values())
    section_matrix = torch.stack(all_section_embs)

    eval_sections = [id for id, content in metadata.items() if content['set'] == 'eval']
    print(f'Eval sections: {len(eval_sections)}')
    print(f'Variants per section: {NUM_VARIANT}')
    print()    

    total_variants = 0
    top1_correct = 0
    top3_correct = 0
    per_section_results = []

    with torch.no_grad():
        for section_id in sorted(eval_sections):
            original_image = Image.open(DATA_DIR / 'sections' / f'{section_id}.jpg')
            section_hits_1 = 0
            section_hits_3 = 0

            for i in range(NUM_VARIANT):
                aug_img = Augmentation()(original_image)
                tensor = image_transform(aug_img).unsqueeze(0)
                emb = head(backbone(tensor)).squeeze(0)

                distance = (section_matrix - emb.unsqueeze(0)).pow(2).sum(dim=1).sqrt()
                topk_indices = torch.argsort(distance)[:3]
                top_ids = [all_section_ids[index] for index in topk_indices]

                if top_ids[0] == section_id:
                    top1_correct += 1
                    section_hits_1 += 1
                if section_id in top_ids:
                    top3_correct += 1
                    section_hits_3 += 1
                total_variants += 1

            accuracy_1 = section_hits_1 / NUM_VARIANT
            accuracy_3 = section_hits_3 / NUM_VARIANT
            per_section_results.append({
                'section_id': section_id,
                'top1': accuracy_1,
                'top3': accuracy_3
            })
            status = 'pass' if accuracy_1 >= 0.75 else 'fail'
            print(f'  {section_id:20s}  top1: {accuracy_1:.1%}  top3: {accuracy_3:.1%}  {status}')

    print()
    overall_top1 = top1_correct / total_variants if total_variants > 0 else 0
    overall_top3 = top3_correct / total_variants if total_variants > 0 else 0
    print(f'Overall top-1 accuracy: {overall_top1: .1%} ({top1_correct}/{total_variants})')
    print(f'Overall top-3 accuracy: {overall_top3: .1%}  ({top3_correct}/{total_variants})')

    log_path = BASE_DIR / 'cached' / 'logs_cached' / f'augmentation_invariance_{experiment}.txt'
    with open(log_path, 'w') as f:
        f.write(f'Experiment {experiment}\n')
        f.write(f'Variants per section: {NUM_VARIANT}\n\n')
        for i in per_section_results:
            f.write(f'{i['section_id']:20s}  top1: {i['top1']:.1%}  top3: {i['top3']:.1%}\n')
        f.write(f"\nOverall top-1: {overall_top1:.1%} ({top1_correct}/{total_variants})\n")
        f.write(f"Overall top-3: {overall_top3:.1%} ({top3_correct}/{total_variants})\n")


if __name__ == '__main__':
    evaluate_augmentation(EXPERIMENT)