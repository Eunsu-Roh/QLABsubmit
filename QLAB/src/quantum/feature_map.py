"""
Feature mapping for quantum states
"""

import pennylane as qml


class StatePreparation:
    """
    양자 상태 인코딩
    
    대회 데이터는 이미 양자 상태이므로 직접 로드
    """
    
    def __init__(self, n_qubits: int):
        """
        Args:
            n_qubits: 큐비트 개수
        """
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
    
    def apply(self, state, wires):
        """
        양자 상태를 회로에 직접 로드
        
        Args:
            state: 양자 상태 벡터 (2^n,)
            wires: 큐비트 인덱스
        """
        qml.StatePrep(state, wires=wires)
    
    def __repr__(self):
        return f"StatePreparation(n_qubits={self.n_qubits})"
