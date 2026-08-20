import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent
metadata_csv_path = BASE_DIR / '..' / 'data' / 'pairs' / 'metadata.csv'
augmented_csv_path = BASE_DIR / '..' / 'data' / 'pairs' / 'augmented_pairs.csv'


def metadata_pairs() -> list[dict]:
    pairs = []
    with open(metadata_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader if row['set'] == 'train']

        for i in range(len(data)):
            for j in range(i+1, len(data)):

                img1 = f"sections/{data[i]['section_id']}.jpg"
                img2 = f"sections/{data[j]['section_id']}.jpg"

                if data[i]['topic'] == data[j]['topic']:
                    pairs.append({'img1': img1, 'img2': img2, 'pair_type': 'topic_positive'})
                    
                elif data[i]['topic'] in data[j]['hard_negative'].split(', ') or \
                    data[j]['topic'] in data[i]['hard_negative'].split(', '):
                    pairs.append({'img1': img1, 'img2': img2, 'pair_type': 'hard_negative'})
            
                elif data[i]['cluster'] == data[j]['cluster']:
                    pairs.append({'img1':img1, 'img2': img2, 'pair_type': 'soft_positive'})

                else:
                    pairs.append({'img1': img1, 'img2': img2, 'pair_type': 'plain_negative'})
    return pairs


def augmented_pairs() -> list[dict]:
    pairs = []
    with open(augmented_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

        for i in range(len(data)):
            pairs.append({
                'img1': f"augmented/{data[i]['section_id']}/{data[i]['variant_id']}.jpg",
                'img2': f"sections/{data[i]['section_id']}.jpg",
                'pair_type': 'augmentation_positive'
            })
            for j in range(i+1, len(data)):
                if data[i]['section_id'] == data[j]['section_id']:
                    pairs.append({
                        'img1': f"augmented/{data[i]['section_id']}/{data[i]['variant_id']}.jpg", 
                        'img2': f"augmented/{data[j]['section_id']}/{data[j]['variant_id']}.jpg",
                        'pair_type': 'augmentation_positive'
                    })
    return pairs


def main():
    merge_list = augmented_pairs() + metadata_pairs()
    pairs_out = BASE_DIR / '..' / 'data' / 'pairs' / 'all_pairs.csv'
    pairs_out.parent.mkdir(parents=True, exist_ok=True)

    with open(pairs_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['img1', 'img2', 'pair_type'])
        writer.writeheader()
        writer.writerows(merge_list)


if __name__ == '__main__':
    main()
