"""
학습 결과 시각화 — 3개 모델의 학습 곡선과 성능 비교 차트 생성
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150

CHECKPOINT_DIR = Path("g:/AIEKG/ml/checkpoints")
OUTPUT_DIR = Path("g:/AIEKG/docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_training_curves():
    """3개 모델의 학습 곡선을 하나의 figure에."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 1. CNN-TCN 3-lead (Arrhythmia Classification) ---
    data = load_json(CHECKPOINT_DIR / "cnn-tcn-3lead/train_results.json")
    h = data["history"]
    epochs = range(1, len(h["train_acc"]) + 1)

    ax = axes[0]
    ax.plot(epochs, h["train_acc"], 'b-', label='Train Acc', linewidth=1.5)
    ax.plot(epochs, h["val_acc"], 'r-', label='Val Acc', linewidth=1.5)
    best = data["best_epoch"]
    ax.axvline(x=best, color='green', linestyle='--', alpha=0.7, label=f'Best (ep {best})')
    ax.axhline(y=data["test_accuracy"], color='orange', linestyle=':', alpha=0.8,
               label=f'Test: {data["test_accuracy"]:.1%}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Arrhythmia Classification\n(CNN-TCN 3-lead, 126K params)')
    ax.legend(fontsize=9)
    ax.set_ylim(0.83, 0.92)
    ax.grid(True, alpha=0.3)

    # --- 2. ResNet-34 + Clinical (Paroxysmal AF Detection) ---
    data2 = load_json(CHECKPOINT_DIR / "paroxysmal-af-resnet34-clinical/results.json")
    h2 = data2["history"]
    epochs2 = range(1, len(h2["val_auroc"]) + 1)

    ax = axes[1]
    ax.plot(epochs2, h2["train_loss"], 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs2, h2["val_loss"], 'r-', label='Val Loss', linewidth=1.5)

    ax2 = ax.twinx()
    ax2.plot(epochs2, h2["val_auroc"], 'g-', label='Val AUROC', linewidth=2)
    best2 = data2["best_epoch"]
    ax2.axhline(y=data2["test_auroc"], color='orange', linestyle=':', alpha=0.8,
                label=f'Test AUROC: {data2["test_auroc"]:.4f}')
    ax.axvline(x=best2, color='green', linestyle='--', alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('AUROC')
    ax.set_title('Paroxysmal AF Detection\n(ResNet-34 + Clinical, 7.3M params)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
    ax.grid(True, alpha=0.3)

    # --- 3. Experiment Comparison (all models) ---
    ax = axes[2]

    experiments = [
        ("CNN-TCN\n12-lead", 93.6, '#2196F3'),
        ("+ BP", 90.3, '#f44336'),
        ("+ CBAM", 90.6, '#f44336'),
        ("CNN-TCN\n3-lead", 90.6, '#4CAF50'),
    ]

    names = [e[0] for e in experiments]
    values = [e[1] for e in experiments]
    colors = [e[2] for e in experiments]

    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Classification Experiments\n(Green = deployed, Red = failed)')
    ax.set_ylim(88, 95)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.15,
                f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'training_curves.png'}")
    plt.close()


def plot_model_comparison():
    """3종 모델 성능 + 논문 대비 비교."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 1. Our 3 models ---
    ax = axes[0]
    models = ['Arrhythmia\nClassification', 'Paroxysmal\nAF Detection', 'AFib\nPrediction']
    metrics = [90.6, 82.4, 73.6]
    colors = ['#4CAF50', '#4CAF50', '#FF9800']
    labels_text = ['90.6% Acc', 'AUROC 0.824', 'AUROC 0.736']

    bars = ax.barh(models, metrics, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5, height=0.5)
    ax.set_xlim(60, 100)
    ax.set_xlabel('Performance (%)')
    ax.set_title('AI EKG — 3 Model Performance')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, label in zip(bars, labels_text):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2.,
                label, ha='left', va='center', fontsize=10, fontweight='bold')

    # --- 2. vs Published Literature ---
    ax = axes[1]

    studies = ['Attia (Mayo)\n12-lead, 600K', 'Raghunath\n12-lead, 400K',
               'Khurshid (MGH)\n12-lead, 100K', 'This Project\n3-lead, 160K']
    aurocs = [0.87, 0.85, 0.81, 0.82]
    colors2 = ['#90CAF9', '#90CAF9', '#90CAF9', '#4CAF50']

    bars = ax.barh(studies, aurocs, color=colors2, alpha=0.8, edgecolor='black', linewidth=0.5, height=0.5)
    ax.set_xlim(0.75, 0.92)
    ax.set_xlabel('AUROC')
    ax.set_title('Paroxysmal AF Detection\nvs Published Studies')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2.,
                f'{val:.2f}', ha='left', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'model_comparison.png'}")
    plt.close()


def plot_confusion_matrix():
    """부정맥 분류 모델 Confusion Matrix."""
    data = load_json(CHECKPOINT_DIR / "cnn-tcn-3lead/train_results.json")
    cm = np.array(data["confusion_matrix"])
    classes = ['Normal', 'AFib', 'Other']

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

    for i in range(3):
        for j in range(3):
            color = 'white' if cm_pct[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm_pct[i,j]:.1f}%\n({cm[i,j]:,})',
                    ha='center', va='center', color=color, fontsize=10)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Arrhythmia Classification — Confusion Matrix\n(CNN-TCN 3-lead, Test Set: 117,182 samples)')

    plt.colorbar(im, ax=ax, label='%', shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.png'}")
    plt.close()


if __name__ == "__main__":
    plot_training_curves()
    plot_model_comparison()
    plot_confusion_matrix()
    print("\nAll charts generated in docs/images/")
