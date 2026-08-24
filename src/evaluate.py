import torch
import csv
from pathlib import Path 
from cache_train import EmbeddingHead

BASE_DIR = Path(__file__).parent / '..'
DATA_DIR = BASE_DIR / 'data'
CACHE_PATH = DATA_DIR / 'features_cache.pt'
METADATA_PATH = DATA_DIR / 'pairs' / 'metadata.csv'
EMBEDDINGS_DIR = DATA_DIR / 'embeddings'

EXPERIMENT = 'B_no_soft'


def generate_embeddings(check_point_path: Path, experiment_name: str):
    cache = torch.load(CACHE_PATH, weights_only=True, map_location='cpu')
    model = EmbeddingHead()
    model.load_state_dict(torch.load(check_point_path, weights_only=True, map_location='cpu'))
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for path, features in cache.items():
            if 'augmented' in path:
                continue
            emb = model(features.unsqueeze(0)).squeeze(0)
            section_id = Path(path).stem
            embeddings[section_id] = emb

    save_path = EMBEDDINGS_DIR / f'embeddings_{experiment_name}.pt'
    torch.save(embeddings, save_path)
    print(f'Saved {len(embeddings)} embeddings to {save_path}')
    return embeddings


def load_metadata():
    metadata = {}
    with open(METADATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['section_id']] = row
    return metadata


def evaluate(embeddings: dict, metadata: dict, k_values=(1, 3, 5)):
    train_ids = []
    train_embs = []
    eval_ids = []
    eval_embs = []
    results_path = BASE_DIR / 'cached' / 'logs_cached' / f'eval_{experiment}.txt'
    f = open(results_path, 'w')

    def log(text=''):
        print(text)
        f.write(text + '\n')

    for sectiond_id, emb in embeddings.items():
        split = metadata[sectiond_id]['set']
        if split == 'train':
            train_ids.append(sectiond_id)
            train_embs.append(emb)
        elif split == 'eval':
            eval_ids.append(sectiond_id)
            eval_embs.append(emb)
    log(f'Train sections: {len(train_ids)}')
    log(f'Eval sections: {len(eval_ids)}')
    log()

    train_matrix = torch.stack(train_embs)
    max_k = max(k_values)

    all_results = []
    for i, (query_id, query_emb) in enumerate(zip(eval_ids, eval_embs)):
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
        log(f'\nQuery: {q['query_id']}  topic: {q['query_topic']}  cluster: {q['query_cluster']}')
        for i, j in enumerate(q['retrieved']):
            match_flag = 'topic' if j['topic_match'] else ('cluster' if j['cluster_match'] else '')
            log(f'  {i+1}. {j['section_id']:12s}  topic: {j['topic']:30s}  dist: {j['distance']:.3f}  {match_flag}')
 
    f.close()
    return all_results


if __name__ == '__main__':
    experiment = EXPERIMENT
    checkpoint = BASE_DIR / 'cached' / 'checkpoints_cached' / f'experiment_{experiment}' / 'best.pt'

    print(f'Evaluating experiment {experiment}')
    print()

    embeddings = generate_embeddings(checkpoint, experiment)
    metadata = load_metadata()
    evaluate(embeddings, metadata)