"""
Test cases for data preprocessing
"""

import unittest
import numpy as np
import torch
import sys
import os

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.preprocessing import (
    normalize_states,
    validate_quantum_states,
    add_noise,
    to_torch_tensors
)


class TestPreprocessing(unittest.TestCase):
    """데이터 전처리 테스트"""
    
    def setUp(self):
        """테스트 데이터 생성"""
        # 랜덤 양자 상태
        self.n_samples = 16
        self.dim = 256
        
        self.states = np.random.randn(self.n_samples, self.dim) + \
                      1j * np.random.randn(self.n_samples, self.dim)
        self.labels = np.random.randint(0, 4, self.n_samples)
    
    def test_normalize_states(self):
        """상태 정규화 테스트"""
        normalized = normalize_states(self.states)
        
        # L2 norm = 1
        norms = np.sqrt(np.sum(np.abs(normalized)**2, axis=1))
        np.testing.assert_array_almost_equal(norms, np.ones(self.n_samples), decimal=5)
    
    def test_validate_quantum_states(self):
        """상태 검증 테스트"""
        # 정규화되지 않은 상태 → False
        is_valid = validate_quantum_states(self.states)
        self.assertFalse(is_valid)
        
        # 정규화된 상태 → True
        normalized = normalize_states(self.states)
        is_valid = validate_quantum_states(normalized)
        self.assertTrue(is_valid)
    
    def test_add_noise(self):
        """노이즈 추가 테스트"""
        normalized = normalize_states(self.states)
        noisy = add_noise(normalized, noise_level=0.01)
        
        # 여전히 정규화되어 있는지
        is_valid = validate_quantum_states(noisy)
        self.assertTrue(is_valid)
        
        # 원본과 다른지
        self.assertFalse(np.allclose(normalized, noisy))
    
    def test_to_torch_tensors(self):
        """PyTorch tensor 변환 테스트"""
        t_X, t_Y = to_torch_tensors(self.states, self.labels)
        
        # 타입 확인
        self.assertIsInstance(t_X, torch.Tensor)
        self.assertIsInstance(t_Y, torch.Tensor)
        
        # dtype 확인
        self.assertEqual(t_X.dtype, torch.complex64)
        self.assertEqual(t_Y.dtype, torch.long)
        
        # Shape 확인
        self.assertEqual(t_X.shape, (self.n_samples, self.dim))
        self.assertEqual(t_Y.shape, (self.n_samples,))


if __name__ == '__main__':
    unittest.main()
