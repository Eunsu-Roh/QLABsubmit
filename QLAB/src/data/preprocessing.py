"""
Data preprocessing utilities
"""

import numpy as np
import torch


def normalize_states(states):
    """
    양자 상태 정규화 (L2 norm = 1)
    
    Args:
        states: (n_samples, 2^n_qubits) complex array
    
    Returns:
        정규화된 상태
    """
    norms = np.sqrt(np.sum(np.abs(states)**2, axis=1, keepdims=True))
    normalized = states / (norms + 1e-10)
    return normalized


def validate_quantum_states(states, tolerance=1e-5):
    """
    양자 상태 유효성 검사
    
    Args:
        states: (n_samples, 2^n_qubits)
        tolerance: 정규화 허용 오차
    
    Returns:
        bool: 모두 유효하면 True
    """
    norms = np.sum(np.abs(states)**2, axis=1)
    
    is_valid = np.allclose(norms, 1.0, atol=tolerance)
    
    if not is_valid:
        print(f"⚠️  Warning: Some states not normalized")
        print(f"   Min norm: {norms.min():.6f}")
        print(f"   Max norm: {norms.max():.6f}")
    
    return is_valid


def to_torch_tensors(train_X, train_Y):
    """
    NumPy arrays를 PyTorch tensors로 변환
    
    Args:
        train_X: (n_samples, 256) complex64
        train_Y: (n_samples,) int64
    
    Returns:
        torch tensors
    """
    t_train_X = torch.tensor(train_X, dtype=torch.complex64)
    t_train_Y = torch.tensor(train_Y, dtype=torch.long)
    
    return t_train_X, t_train_Y
