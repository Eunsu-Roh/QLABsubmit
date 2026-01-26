"""
Test cases for VQC classifier
"""

import unittest
import torch
import sys
import os

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import VQCClassifier


class TestVQCClassifier(unittest.TestCase):
    """VQCClassifier 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.model = VQCClassifier(
            n_qubits=8,
            n_layers=3,
            topology='linear',
            measurement_qubits=[6, 7]
        )
    
    def test_model_initialization(self):
        """모델 초기화 테스트"""
        self.assertIsNotNone(self.model.qc)
        self.assertIsNotNone(self.model.params)
        self.assertTrue(self.model.params.requires_grad)
    
    def test_forward_single_sample(self):
        """단일 샘플 forward 테스트"""
        # 단일 양자 상태
        state = torch.randn(256, dtype=torch.complex64)
        state = state / torch.norm(state)
        
        # Forward
        output = self.model(state)
        
        # Shape 확인
        self.assertEqual(output.shape, (4,))
        
        # 확률 합 = 1
        self.assertAlmostEqual(output.sum().item(), 1.0, places=5)
    
    def test_forward_batch(self):
        """배치 forward 테스트"""
        # 배치 양자 상태
        batch_size = 4
        states = torch.randn(batch_size, 256, dtype=torch.complex64)
        
        # 정규화
        for i in range(batch_size):
            states[i] = states[i] / torch.norm(states[i])
        
        # Forward
        output = self.model(states)
        
        # Shape 확인
        self.assertEqual(output.shape, (batch_size, 4))
        
        # 각 샘플의 확률 합 = 1
        for i in range(batch_size):
            self.assertAlmostEqual(output[i].sum().item(), 1.0, places=5)
    
    def test_parameter_gradient(self):
        """파라미터 gradient 테스트"""
        # 랜덤 입력
        state = torch.randn(256, dtype=torch.complex64)
        state = state / torch.norm(state)
        
        # Forward + Loss
        output = self.model(state)
        loss = -torch.log(output[0])
        
        # Backward
        loss.backward()
        
        # Gradient 존재 확인
        self.assertIsNotNone(self.model.params.grad)
        self.assertTrue(torch.any(self.model.params.grad != 0))


if __name__ == '__main__':
    unittest.main()
