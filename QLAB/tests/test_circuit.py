"""
Test cases for quantum circuit
"""

import unittest
import torch
import numpy as np
import sys
import os

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum import QuantumCircuit, StatePreparation, VariationalAnsatz


class TestQuantumCircuit(unittest.TestCase):
    """양자 회로 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.n_qubits = 8
        self.n_layers = 3
        self.topology = 'linear'
        self.measurement_qubits = [6, 7]
    
    def test_state_preparation(self):
        """StatePreparation 테스트"""
        feature_map = StatePreparation(self.n_qubits)
        
        self.assertEqual(feature_map.n_qubits, self.n_qubits)
        self.assertEqual(feature_map.state_dim, 2**self.n_qubits)
    
    def test_ansatz_parameters(self):
        """Ansatz 파라미터 개수 테스트"""
        ansatz = VariationalAnsatz(self.n_qubits, self.n_layers, self.topology)
        
        expected_params = self.n_layers * 3 * self.n_qubits
        self.assertEqual(ansatz.n_params, expected_params)
    
    def test_circuit_output_shape(self):
        """회로 출력 shape 테스트"""
        qc = QuantumCircuit(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            topology=self.topology,
            measurement_qubits=self.measurement_qubits
        )
        
        # 랜덤 양자 상태
        state = torch.randn(2**self.n_qubits, dtype=torch.complex64)
        state = state / torch.norm(state)
        
        # 랜덤 파라미터
        params = torch.randn(qc.ansatz.n_params)
        
        # Forward pass
        output = qc.forward(state, params)
        
        # 2-qubit 측정 → 4개 확률
        self.assertEqual(output.shape, (4,))
        
        # 확률 합 = 1
        self.assertAlmostEqual(output.sum().item(), 1.0, places=5)
    
    def test_different_topologies(self):
        """다양한 토폴로지 테스트"""
        topologies = ['linear', 'ring', 'bidirectional']
        
        for topology in topologies:
            ansatz = VariationalAnsatz(self.n_qubits, self.n_layers, topology)
            self.assertEqual(ansatz.topology, topology)
    
    def test_invalid_topology(self):
        """잘못된 토폴로지 에러 테스트"""
        with self.assertRaises(ValueError):
            VariationalAnsatz(self.n_qubits, self.n_layers, 'invalid')
    
    def test_measurement_qubits_validation(self):
        """측정 큐비트 검증 테스트"""
        # 정상 케이스
        qc = QuantumCircuit(measurement_qubits=[0, 1])
        self.assertEqual(qc.measurement_qubits, [0, 1])
        
        # 에러 케이스: 1개만
        with self.assertRaises(ValueError):
            QuantumCircuit(measurement_qubits=[0])
        
        # 에러 케이스: 범위 초과
        with self.assertRaises(ValueError):
            QuantumCircuit(measurement_qubits=[0, 10])


if __name__ == '__main__':
    unittest.main()
