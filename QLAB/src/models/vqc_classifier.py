"""
Variational Quantum Classifier
"""

import torch
import torch.nn as nn
from ..quantum.circuit import QuantumCircuit


class VQCClassifier(nn.Module):
    """
    PyTorch wrapper for Variational Quantum Circuit
    
    고전 optimizer와 양자 회로를 연결
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 5,
        topology: str = 'linear',
        measurement_qubits: list = [6, 7],
        ansatz_type: str = 'variational',
        init_scale: float = 0.01,
        measurement_type: str = 'computational'
    ):
        """
        Args:
            n_qubits: 큐비트 개수
            n_layers: ansatz 레이어 개수
            topology: 얽힘 토폴로지 (variational ansatz용)
            measurement_qubits: 측정 큐비트
            ansatz_type: ansatz 종류 ('variational', 'sel', 'hea', 'alternating', 'two_local',
                         'hybrid', 'alt_hybrid', 'balanced', 'qcnn')
            init_scale: 파라미터 초기화 스케일
            measurement_type: 측정 방식 ('computational' or 'pauli')
        """
        super().__init__()
        
        self.measurement_type = measurement_type
        
        # Quantum circuit
        self.qc = QuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            topology=topology,
            measurement_qubits=measurement_qubits,
            ansatz_type=ansatz_type,
            measurement_type=measurement_type
        )
        
        # 학습 가능한 파라미터
        self.n_params = self.qc.ansatz.n_params
        self.params = nn.Parameter(
            torch.randn(self.n_params, requires_grad=True) * init_scale
        )
        
        # Pauli 측정용 선형 변환 레이어 (10 -> 4)
        if measurement_type == 'pauli':
            self.pauli_to_probs = nn.Linear(10, 4)
            # Softmax 출력을 위한 초기화
            nn.init.xavier_uniform_(self.pauli_to_probs.weight)
            nn.init.zeros_(self.pauli_to_probs.bias)
        
        print(f"\n🧠 VQCClassifier created")
        print(f"   Parameters: {self.n_params}")
        print(f"   Init scale: {init_scale}")
        if measurement_type == 'pauli':
            print(f"   Pauli->Probs layer: 10 -> 4")
        print(f"   Param mean: {self.params.mean().item():.6f}")
        print(f"   Param std: {self.params.std().item():.6f}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 배치 양자 상태 (batch_size, 256)
        
        Returns:
            확률 분포 (batch_size, 4)
        """
        output = self.qc.forward(x, self.params)
        
        if self.measurement_type == 'pauli':
            # Pauli 측정값 (batch, 6) -> 확률 (batch, 4)
            logits = self.pauli_to_probs(output)
            probs = torch.softmax(logits, dim=-1)
            return probs
        else:
            # Computational basis는 이미 확률
            return output
    
    def get_quantum_circuit(self):
        """양자 회로 객체 반환"""
        return self.qc
    
    def __repr__(self):
        return f"VQCClassifier(n_params={self.n_params}, {self.qc})"
