"""
Loss functions for quantum classification
"""

import torch
import torch.nn.functional as F


def cross_entropy_loss(probs, labels, class_weights=None, smoothing=0.0):
    """
    Cross-Entropy Loss
    
    Args:
        probs: (batch_size, n_classes) 확률 분포
        labels: (batch_size,) 정답 라벨
        class_weights: (n_classes,) 클래스별 가중치 (optional)
    
    Returns:
        scalar loss
    """
    # One-hot encoding with label smoothing
    n_classes = probs.shape[1]
    labels_onehot = F.one_hot(labels, num_classes=n_classes).float()
    if smoothing > 0.0:
        labels_onehot = labels_onehot * (1 - smoothing) + smoothing / n_classes
    # Normalize probabilities
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    # Cross-entropy: -Σ y_true * log(y_pred)
    loss = -torch.sum(labels_onehot * torch.log(probs + 1e-10), dim=1)
    # Apply class weights
    if class_weights is not None:
        weights = class_weights[labels]
        loss = loss * weights
    return torch.mean(loss)


def focal_loss(probs, labels, gamma=2.0, alpha=None):
    """
    Focal Loss - hard sample에 집중
    
    Args:
        probs: (batch_size, n_classes)
        labels: (batch_size,)
        gamma: focusing parameter (default: 2.0)
        alpha: class weights (optional)
    
    Returns:
        scalar loss
    """
    labels_onehot = F.one_hot(labels, num_classes=4).float()
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    
    # 정답 클래스 확률
    p_correct = torch.sum(probs * labels_onehot, dim=1)
    
    # Focal weight: (1-p)^γ
    focal_weight = (1 - p_correct) ** gamma
    
    # Class weights (optional)
    if alpha is not None:
        alpha_t = alpha[labels]
        focal_weight = alpha_t * focal_weight
    
    loss = -focal_weight * torch.log(p_correct + 1e-10)
    
    return torch.mean(loss)


def get_loss_function(loss_name='ce', **kwargs):
    """
    손실 함수 팩토리
    
    Args:
        loss_name: 'ce' or 'focal'
        **kwargs: 손실 함수별 추가 인자
            - class_weights: (n_classes,) tensor for CE loss
            - gamma, alpha: focal loss parameters
    
    Returns:
        loss function
    """
    if loss_name == 'ce':
        class_weights = kwargs.get('class_weights', None)
        smoothing = kwargs.get('smoothing', 0.0)
        return lambda probs, labels: cross_entropy_loss(probs, labels, class_weights, smoothing)
    elif loss_name == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', None)
        return lambda probs, labels: focal_loss(probs, labels, gamma, alpha)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
