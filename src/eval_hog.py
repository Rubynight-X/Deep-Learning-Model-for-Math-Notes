import csv
import numpy as np
import torch
from PIL import Image, ImageOps
from pathlib import Path
from skimage.feature import hog
from data_loader import pad_to_square

BASE_DIR = Path(__file__).parent / '..'
DATA_DIR = BASE_DIR / 'data'
METADATA_PATH = DATA_DIR / 'pairs' / 'metadata.csv'


def compute_hog(image: Image.Image) -> np.ndarray:
    """Compute HOG descriptor from a PIL image."""
    img = pad_to_square(image)
    img_gray = np.array(img.convert('L'))

    descriptor = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # L2 normalize for cosine similarity via Euclidean distance
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm

    return descriptor


def load_metadata():
    metadata = {}
    with open(METADATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['section_id']] = row
    return metadata


def generate_hog_embeddings(metadata):
    """Compute HOG descriptors for all sections."""
    embeddings = {}
    for section_id in sorted(metadata.keys()):
        img_path = DATA_DIR / 'sections' / f'{section_id}.jpg'
        image = Image.open(img_path)
        descriptor = compute_hog(image)
        embeddings[section_id] = torch.tensor(descriptor, dtype=torch.float32)

    print(f"Computed HOG descriptors for {len(embeddings)} sections")
    print(f"Descriptor dimension: {list(embeddings.values())[0].shape[0]}")
    return embeddings


def evaluate(embeddings: dict, metadata: dict, k_values=(1, 3, 5)):
    train_ids = []
    train_embs = []
    eval_ids = []
    eval_embs = []

    results_path = BASE_DIR / 'slow' / 'logs_slow' / 'eval_HOG.txt'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(results_path, 'w')

    def log(text=''):
        print(text)
        f.write(text + '\n')

    for section_id, emb in embeddings.items():
        split = metadata[section_id]['set']
        if split == 'train':
            train_ids.append(section_id)
            train_embs.append(emb)
        elif split == 'eval':
            eval_ids.append(section_id)
            eval_embs.append(emb)

    log(f'Train sections: {len(train_ids)}')
    log(f'Eval sections: {len(eval_ids)}')
    log()

    train_matrix = torch.stack(train_embs)
    max_k = max(k_values)

    all_results = []
    for query_id, query_emb in zip(eval_ids, eval_embs):
        query_topic = metadata[query_id]['topic']
        query_cluster = metadata[query_id]['cluster']

        distance = (train_matrix - query_emb.unsqueeze(0)).pow(2).sum(dim=1).sqrt()
        topk_indices = torch.argsort(distance)[:max_k]

        retrieved = []
        for index in topk_indices:
            retrieve_id = train_ids[index]
            retrieve_topic = metadata[retrieve_id]['topic']
            retrieve_cluster = metadata[retrieve_id]['cluster']
            retrieve_distance = distance[index].item()
            retrieved.append({
                'section_id': retrieve_id,
                'topic': retrieve_topic,
                'cluster': retrieve_cluster,
                'distance': retrieve_distance,
                'topic_match': retrieve_topic == query_topic,
                'cluster_match': retrieve_cluster == query_cluster
            })

        all_results.append({
            'query_id': query_id,
            'query_topic': query_topic,
            'query_cluster': query_cluster,
            'retrieved': retrieved
        })

    log('PRECISION@K (topic match)')
    for k in k_values:
        topic_hits = 0
        cluster_hits = 0
        total = 0
        topic_match = 0
        cluster_match = 0
        for result in all_results:
            top_k = result['retrieved'][:k]
            if any(i['topic_match'] for i in top_k):
                topic_match += 1
            if any(i['cluster_match'] for i in top_k):
                cluster_match += 1
            topic_hits += sum(1 for i in top_k if i['topic_match'])
            cluster_hits += sum(1 for i in top_k if i['cluster_match'])
            total += k
        topic_precision = topic_hits / total if total > 0 else 0
        cluster_precision = cluster_hits / total if total > 0 else 0
        topic_hit_rate = topic_match / len(all_results)
        cluster_hit_rate = cluster_match / len(all_results)
        log(f'  precision@{k}  topic_precision: {topic_precision: .3f}  topic_hit_rate: {topic_hit_rate: .3f}  cluster_precision: {cluster_precision: .3f}  cluster_hit_rate: {cluster_hit_rate: .3f}')

    log()
    log('per-query retrieval results (top-{})'.format(max_k))
    for result in all_results:
        q = result
        log(f"\nQuery: {q['query_id']}  topic: {q['query_topic']}  cluster: {q['query_cluster']}")
        for i, j in enumerate(q['retrieved']):
            match_flag = 'topic' if j['topic_match'] else ('cluster' if j['cluster_match'] else '')
            log(f"  {i+1}. {j['section_id']:12s}  topic: {j['topic']:30s}  dist: {j['distance']:.3f}  {match_flag}")

    f.close()
    print(f"\nSaved to {results_path}")
    return all_results


if __name__ == '__main__':
    print("HOG Baseline Evaluation")
    print()
    metadata = load_metadata()
    embeddings = generate_hog_embeddings(metadata)
    evaluate(embeddings, metadata)