"""
Visualization utilities
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .metrics import calculate_confusion_matrix


PHASE_NAMES = ['Cluster', 'Trivial', 'Ferro', 'Anti-F']


def plot_training_history(loss_history, acc_history, save_path=None):
    """
    훈련 history 시각화
    
    Args:
        loss_history: list of loss values
        acc_history: list of accuracy values
        save_path: 저장 경로 (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(loss_history) + 1)
    
    # Loss curve
    axes[0].plot(epochs, loss_history, 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Accuracy curve
    axes[1].plot(epochs, acc_history, 'g-', linewidth=2, label='Train Acc')
    axes[1].axhline(y=0.25, color='r', linestyle='--', alpha=0.5, label='Random (25%)')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 그래프 저장: {save_path}")
    
    plt.show()


def plot_confusion_matrix(true_labels, pred_labels, save_path=None):
    """
    Confusion Matrix 시각화
    
    Args:
        true_labels: ground truth labels
        pred_labels: predicted labels
        save_path: 저장 경로 (optional)
    """
    cm = calculate_confusion_matrix(true_labels, pred_labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=PHASE_NAMES, yticklabels=PHASE_NAMES,
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Confusion Matrix 저장: {save_path}")
    
    plt.show()


def plot_parameter_distribution(model, save_path=None):
    """
    학습된 파라미터 분포 시각화
    
    Args:
        model: VQCClassifier
        save_path: 저장 경로 (optional)
    """
    params = model.params.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(params, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(params.mean(), color='r', linestyle='--', 
                    linewidth=2, label=f'Mean: {params.mean():.4f}')
    axes[0].set_xlabel('Parameter Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Parameter Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Parameter evolution
    axes[1].plot(params, 'o-', alpha=0.5, markersize=3)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Parameter Index', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Parameter Values', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 파라미터 분포 저장: {save_path}")
    
    plt.show()


def plot_probability_heatmap(pred_probs, true_labels, save_path=None):
    """
    예측 확률 히트맵
    
    Args:
        pred_probs: (n_samples, 4) prediction probabilities
        true_labels: ground truth labels
        save_path: 저장 경로 (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 정렬 (true label 순서대로)
    sorted_idx = np.argsort(true_labels)
    sorted_probs = pred_probs[sorted_idx]
    sorted_labels = true_labels[sorted_idx]
    
    sns.heatmap(sorted_probs.T, cmap='YlOrRd', ax=ax,
                yticklabels=PHASE_NAMES, cbar_kws={'label': 'Probability'})
    
    ax.set_xlabel('Sample Index (sorted by true label)', fontsize=12)
    ax.set_ylabel('Predicted Class', fontsize=12)
    ax.set_title('Prediction Probability Heatmap', fontsize=14, fontweight='bold')
    
    # 구분선 추가
    for i in range(1, 4):
        boundary = np.where(sorted_labels == i)[0][0] if i in sorted_labels else len(sorted_labels)
        ax.axvline(boundary, color='blue', linewidth=2, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 확률 히트맵 저장: {save_path}")
    
    plt.show()
