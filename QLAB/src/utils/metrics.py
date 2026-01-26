"""
Evaluation metrics
"""

import torch
import numpy as np


PHASE_NAMES = ['Cluster(SPT)', 'Trivial', 'Ferro', 'Anti-F']


def evaluate_model(model, train_X, train_Y, device='cpu', verbose=True):
    """
    모델 평가
    
    Args:
        model: VQCClassifier
        train_X: torch.Tensor (n_samples, 256)
        train_Y: torch.Tensor (n_samples,)
        device: 'cpu' or 'cuda'
        verbose: 상세 출력 여부
    
    Returns:
        pred_labels, pred_probs, accuracy
    """
    model.eval()
    
    with torch.no_grad():
        predictions = model(train_X.to(device))
        pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        pred_probs = predictions.cpu().numpy()
    
    true_labels = train_Y.numpy()
    accuracy = (true_labels == pred_labels).mean()
    
    if verbose:
        print(f"\n{'='*70}")
        print("  평가 결과")
        print(f"{'='*70}")
        print(f"   전체 정확도: {accuracy*100:.2f}%\n")
        
        _print_sample_results(true_labels, pred_labels, pred_probs)
        _print_class_accuracy(true_labels, pred_labels)
    
    return pred_labels, pred_probs, accuracy


def _print_sample_results(true_labels, pred_labels, pred_probs):
    """샘플별 예측 결과 출력"""
    print(f"{'Sample':>6} {'True':>12} {'Pred':>12} {'Confidence':>12} {'Status':>8}")
    print("-" * 60)
    
    for i in range(len(true_labels)):
        true_name = PHASE_NAMES[true_labels[i]]
        pred_name = PHASE_NAMES[pred_labels[i]]
        confidence = pred_probs[i, pred_labels[i]]
        status = "✅" if true_labels[i] == pred_labels[i] else "❌"
        
        print(f"{i:6d} {true_name:>12} {pred_name:>12} {confidence:11.2%} {status:>8}")


def _print_class_accuracy(true_labels, pred_labels):
    """클래스별 정확도 출력"""
    print(f"\n{'='*70}")
    print("  클래스별 성능")
    print(f"{'='*70}")
    
    for label in range(4):
        mask = true_labels == label
        if mask.sum() > 0:
            class_acc = (pred_labels[mask] == true_labels[mask]).mean()
            print(f"   {PHASE_NAMES[label]:15s}: {class_acc*100:5.1f}% ({mask.sum()}개 샘플)")


def calculate_class_accuracy(true_labels, pred_labels):
    """
    클래스별 정확도 계산
    
    Returns:
        dict: {class_label: accuracy}
    """
    class_acc = {}
    
    for label in range(4):
        mask = true_labels == label
        if mask.sum() > 0:
            class_acc[label] = (pred_labels[mask] == true_labels[mask]).mean()
        else:
            class_acc[label] = None
    
    return class_acc


def calculate_confusion_matrix(true_labels, pred_labels):
    """
    Confusion matrix 계산
    
    Returns:
        np.ndarray (4, 4)
    """
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(true_labels, pred_labels)
