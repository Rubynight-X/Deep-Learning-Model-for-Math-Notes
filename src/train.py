import time
import torch
import torch.nn as nn
import data_loader
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path

class MathEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, unfreeze_layer4=False):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.unfreeze_layer4 = unfreeze_layer4

        for param in self.backbone.parameters():
            param.requires_grad = False
        if self.unfreeze_layer4:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        self.embedding_head = nn.Linear(512, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def train(self, mode=True):
        super().train(mode)
        self.backbone.conv1.eval()
        self.backbone.bn1.eval()
        self.backbone.layer1.eval()
        self.backbone.layer2.eval()
        self.backbone.layer3.eval()
        if not self.unfreeze_layer4:
            self.bone.layer4.eval()
        return self


BACKBONE_LEARNING_RATE = 1e-4
HEAD_LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
MARGIN = 1.0
EMBEDDING_DIM = 128

EXPERIMENT = 'C'
EXPERIMENT_CONFIGS = {
    'A': {
        'type': 'augmentation_only',
        'unfreeze': False,
        'description': 'Baseline - augmentation + plain negatives'
    },
     'B': {
        'type': 'all_pairs',
        'unfreeze': False,
        'description': 'All pairs — augmentation + plain negatives + semantic'
    },
    'B_no_soft': {
        'type': 'no_soft_positive',
        'unfreeze': False,
        'description': 'All pairs except soft_positive'
    },
    'C': {
        'type': 'all_pairs',
        'unfreeze': True,
        'description': 'All pairs + unfrozen layer4'
    }
}



def weighted_contrastive_loss(emb_a, emb_b, weight, margin=MARGIN):
    distance = (emb_a - emb_b).pow(2).sum(dim=1).sqrt()
    pull = weight * distance.pow(2)
    push = (1 - weight) * torch.clamp(margin - distance, min=0).pow(2)
    return (pull + push).mean()


def train(experiment=EXPERIMENT):
    config = EXPERIMENT_CONFIGS[experiment]
    print(f"Experiment {experiment}: {config['description']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    model = MathEmbeddingModel(embedding_dim=EMBEDDING_DIM, unfreeze_layer4=config['unfreeze']).to(device)

    if config['unfreeze']:
        optimizer = torch.optim.Adam([
            {'params': model.backbone.layer4.parameters(), 'lr': BACKBONE_LEARNING_RATE},
            {'params': model.embedding_head.parameters(), 'lr': HEAD_LEARNING_RATE}
        ])
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=HEAD_LEARNING_RATE)

    if config['type'] == 'augmentation_only':
        dataset = data_loader.PairDataset(augmentation_only=True, no_soft_positive=False)
    elif config['type'] == 'no_soft_positive':
        dataset = data_loader.PairDataset(augmentation_only=False, no_soft_positive=True)
    else:
        dataset = data_loader.PairDataset(augmentation_only=False, no_soft_positive=False)
    dataset.resample()
    loader = data_loader.build_dataloader(dataset)
    print(f'Pairs after exclusion: {len(dataset)}')
    print(f'Batches per epoch: {len(loader)}')
    print()


    base_dir = Path(__file__).parent / '..'
    checkpoint_dir = base_dir / 'slow' / 'checkpoints_slow' / f"experiment_{experiment}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = base_dir / 'slow' / 'logs_slow'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'experiment_{experiment}.csv'
    with open(log_path, 'w') as f:
        f.write('epoch,avg_loss,avg_pos_dist,avg_neg_dist,elapsed_time\n')

    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        dataset.resample()
        model.train()

        epoch_loss = 0.0
        pos_distances = []
        neg_distances = []
        t0 = time.time()

        for img1, img2, weight in loader:
            img1, img2, weight = img1.to(device), img2.to(device), weight.to(device)
            emb1 = model(img1)
            emb2 = model(img2)
            loss = weighted_contrastive_loss(emb1, emb2, weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * img1.size(0)
            with torch.no_grad():
                dists = (emb1 - emb2).pow(2).sum(dim=1).sqrt()
                pos_mask = weight > 0.0
                neg_mask = weight == 0.0
                if pos_mask.any():
                    pos_distances.append(dists[pos_mask].mean().item())
                if neg_mask.any():
                    neg_distances.append(dists[neg_mask].mean().item())

        elapsed_time = time.time() - t0
        avg_loss = epoch_loss / len(dataset)
        avg_pos_dist = sum(pos_distances) / len(pos_distances) if pos_distances else 0.0
        avg_neg_dist = sum(neg_distances) / len(neg_distances) if neg_distances else 0.0

        print(
            f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"loss {avg_loss:.4f} | "
            f"pos_dist {avg_pos_dist:.3f} | "
            f"neg_dist {avg_neg_dist:.3f} | "
            f"{elapsed_time:.1f}s"
        )

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{avg_loss:.6f},{avg_pos_dist:.4f},{avg_neg_dist:.4f},{elapsed_time:.1f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_dir / 'best.pt')

        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            torch.save(model.state_dict(), checkpoint_dir / f'epoch_{epoch}.pt')


    torch.save(model.state_dict(), checkpoint_dir / 'final.pt')
    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f'    Checkpoints: {checkpoint_dir}')
    print(f'    Training log: {log_path}')


if __name__ == '__main__':
    train()