# MathLens

**Retrieving related sections of handwritten math notes with metric-learned CNN embeddings.**

MathLens is a deep learning project that embeds sections of my own handwritten Calculus and Linear Algebra notes into a vector space, so that searching for one section surfaces related ones. It compares a hand-crafted baseline (HOG), frozen and fine-tuned ResNet18 features, and a custom weighted contrastive loss. It also asks an honest question: *when does visual similarity stop being semantic similarity?*

## Highlights

- **Custom dataset:** 175 hand-cropped sections (86 Calculus, 89 Linear Algebra) from ~40 pages, with a topic / topic-family / hard-negative labeling scheme I designed, and a leakage-free page-level train/eval split.
- **Preprocessing pipeline:** grayscale, deskew, CLAHE, and denoise, built and debugged with OpenCV (including a `minAreaRect` angle-flip bug).
- **Custom weighted contrastive loss** with graded pair weights (topic positives, soft family positives, hand-curated hard negatives), chosen over triplet loss for small-data stability.
- **Feature caching:** caching frozen-backbone features cut epoch time from ~500 s on a Colab T4 GPU to under 1 s on a laptop CPU with no dedicated GPU, with confirmed-equivalent results.
- **Controlled experiments:** HOG baseline, frozen ResNet18 (A, B, B_no_soft), and a fine-tuned `layer4` (C), plus ablations on sampling caps, pair weights, and checkpoints.
- **Streamlit retrieval demo** (`src/user_interface.py`).

## Results

Retrieval on 35 held-out query sections against 140 training sections. "Hit@k" means at least one correct match in the top k; a cluster match means the same topic family.

| Method | Topic hit@5 | Cluster hit@1 | Cluster hit@5 | Augmentation invariance (top-1) |
|---|---|---|---|---|
| HOG baseline | 0.086 | 0.057 | 0.200 | n/a |
| A: frozen ResNet18, augmentation pairs | 0.143 | 0.086 | 0.343 | ~61% |
| B: frozen + semantic labels | 0.086 | 0.171 | 0.343 | ~43% |
| C: unfrozen `layer4` (epoch 20) | 0.171 | 0.143 | 0.286 | n/a |

**What I found:**

1. Pretrained ImageNet features beat HOG, but the frozen backbone groups notes by *visual layout* (theorem blocks, worked examples) rather than mathematical content, so Calculus queries often return Linear Algebra pages.
2. With only 53 topic-positive pairs, the hand-labeled signal was too sparse to reshape that space. It improved family-level retrieval (cluster hit@1 doubled from A to B) at a cost in augmentation invariance.
3. Fine-tuning gave the best topic retrieval but overfit quickly (epoch 20 beat epoch 30).

The absolute numbers are low and the eval set is small (one query is worth ~0.03), so treat the trends as suggestive. The eval set also served as the validation set, and there is no separate held-out test set. The full report covers the details and limitations.

## Quick start

```bash
git clone https://github.com/Rubynight-X/Deep-Learning-Model-for-Math-Notes
cd mathlens
pip install torch torchvision opencv-python pillow numpy matplotlib streamlit
cd src
```

```bash
python preprocess.py          # grayscale, deskew, CLAHE, denoise
python augment.py             # augment training sections
python pair_generation.py     # build the labeled pair pool
python cache_train.py         # train frozen-backbone models (A, B, B_no_soft)
python train.py               # train with unfrozen layer4 (C)
python eval_sementics.py      # topic / cluster retrieval metrics
python eval_augmentation.py   # augmentation invariance
python eval_hog.py            # HOG baseline
streamlit run user_interface.py
```

The note images are personal and not included in this repo. The code and label files are, so the pipeline can be re-run on your own notes.

## Repository structure

```
src/
├── preprocess.py           # grayscale → deskew → CLAHE → denoise
├── augment.py              # training-set augmentation
├── pair_generation.py      # builds the labeled pair pool
├── data_loader.py          # PairDataset and DataLoader (raw images)
├── cache_data_loader.py    # PairDataset over cached backbone features
├── train.py                # full-pipeline training (unfrozen backbone)
├── cache_train.py          # fast training on cached features
├── eval_augmentation.py    # Metric 1: augmentation invariance
├── eval_sementics.py       # Metric 2: topic / cluster retrieval
├── eval_hog.py             # HOG baseline
├── loss_visualization.py   # training-curve plots
└── user_interface.py       # Streamlit demo
docs/
└── REPORT.md               # full methodology, experiments, and findings
```

## Scope and limitations

MathLens does **not** understand mathematics. It measures whether learned *visual* features capture useful structure in handwritten notes. Semantic similarity (via OCR or multimodal models) is future work, along with a deeper embedding head, coarser topic labels to increase supervision, and a larger corpus with FAISS search.

## Full report

See [`docs/REPORT.md`](docs/REPORT.md) for the dataset design, loss derivation, experiment configurations, complete results and ablation tables, failure analysis, and limitations.
