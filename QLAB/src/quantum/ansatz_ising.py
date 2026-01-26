"""
Ising Hamiltonian Ansatz

Transverse Field Ising Model (TFIM)의 시간 진화 연산자에서 영감을 받은 구조.
RZZ (Interaction) 게이트와 RX (Field) 게이트로 구성됨.
"""

import pennylane as qml


class IsingAnsatz:
    """
    Ising Model 구조를 반영한 Ansatz
    
    구조 (per layer):
    1. RZZ gates on nearest neighbors (Ising interaction)
    2. RX gates on all qubits (Transverse field)
    """
    
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 파라미터 개수:
        # Layer당: (n_qubits - 1)개의 RZZ + n_qubits개의 RX
        # = 2 * n_qubits - 1
        self.n_params = n_layers * (2 * n_qubits - 1)
    
    def apply(self, params):
        """
        Args:
            params: flattened parameters
        """
        param_idx = 0
        
        for layer in range(self.n_layers):
            # 1. Ising Interaction (RZZ)
            # 짝수/홀수 패턴으로 나누어 병렬 실행 가능성을 높임 (선택적)
            # 여기서는 단순 선형 연결
            for i in range(self.n_qubits - 1):
                # IsingZZ gate: exp(-i * theta/2 * Z \otimes Z)
                qml.IsingZZ(params[param_idx], wires=[i, i + 1])
                param_idx += 1
            
            # 2. Transverse Field (RX)
            for i in range(self.n_qubits):
                qml.RX(params[param_idx], wires=i)
                param_idx += 1
    
    def __repr__(self):
        return f"IsingAnsatz(n_qubits={self.n_qubits}, n_layers={self.n_layers})"