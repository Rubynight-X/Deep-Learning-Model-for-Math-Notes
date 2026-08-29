import csv
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent / '..'

def load_log(log_path: Path):
    epoch_num = []
    avg_loss = []
    avg_positive_distance = []
    avg_negative_distance = []
    with open(log_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_num.append(int(row['epoch']))
            avg_loss.append(float(row['avg_loss']))
            avg_positive_distance.append(float(row['avg_pos_dist']))
            avg_negative_distance.append(float(row['avg_neg_dist']))
    return epoch_num, avg_loss, avg_positive_distance, avg_negative_distance


def plot(log_path, experiment_name, output_dir):
    epoch, loss, pos_distance, neg_distance = load_log(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch, loss, color='tab:blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average loss')
    ax.set_title(f'Experiment {experiment_name} - Training loss')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{experiment_name}_loss.jpg', dpi=150)
    plt.close(fig)

    # positive distance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch, pos_distance, color='tab:green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Positive Distance')
    ax.set_title(f'Experiment {experiment_name} - Positive Pair Distance')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{experiment_name}_pos_dist.jpg', dpi=150)
    plt.close(fig)

    # negative distance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch, neg_distance, color='tab:red')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Margin (1.0)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Negative Distance')
    ax.set_title(f'Experiment {experiment_name} - Negative Pair Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{experiment_name}_neg_dist.jpg', dpi=150)
    plt.close(fig)


def plot_comparison(logs: dict, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, log_path in logs.items():
        epoch, loss, _, _ = load_log(log_path)
        ax.plot(epoch, loss, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_title('Training Loss - All Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'comparison_loss.jpg', dpi=150)
    plt.close(fig)

    # positive distance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, log_path in logs.items():
        epoch, _, pos_distance, _ = load_log(log_path)
        ax.plot(epoch, pos_distance, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Positive Distance')
    ax.set_title('Positive Pair Distance - All Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'comparison_pos_dist.jpg', dpi=150)
    plt.close(fig)

    # negative distance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, log_path in logs.items():
        epoch, _, _, neg_distance = load_log(log_path)
        ax.plot(epoch, neg_distance, label=name)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Margin (1.0)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Negative Distance')
    ax.set_title('Negative Pair Distance - All Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'comparison_neg_dist.jpg', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    logs = {
        'A_reduced_cap': BASE_DIR / 'cached' / 'logs_cached' / 'experiment_A_reduced_cap.csv',
        'B_reduced_cap': BASE_DIR / 'cached' / 'logs_cached' / 'experiment_B_reduced_cap.csv',
        'B_no_soft_reduced_cap': BASE_DIR / 'cached' / 'logs_cached' / 'experiment_B_no_soft_reduced_cap.csv',
        'C_reduced_cap': BASE_DIR / 'slow' / 'logs_slow' / 'experiment_C_reduced_cap.csv',
        'C_raised_weight': BASE_DIR / 'slow' / 'logs_slow' / 'experiment_C_raised_weight.csv'
    }

    output_dir = BASE_DIR / 'figures'
    for name, log_path in logs.items():
        if log_path.exists():
            plot(log_path, name, output_dir)

    logs = {key: value for key, value in logs.items() if value.exists()}
    if len(logs) >= 2:
        plot_comparison(logs, output_dir)
