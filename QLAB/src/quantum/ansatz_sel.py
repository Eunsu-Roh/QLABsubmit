"""
StronglyEntanglingLayers Ansatz
"""

import pennylane as qml


class StronglyEntanglingAnsatz:
    """
    StronglyEntanglingLayers 기반 Ansatz
    
    특징:
    - 모든 큐비트 간 강한 얽힘
    - Barren plateau 완화
    - 효율적 파라미터 사용
    
    참고:
    - Nature Communications 2018
    - PennyLane StronglyEntanglingLayers
    """
    
    def __init__(self, n_qubits=8, n_layers=4):
        """
        Args:
            n_qubits: 큐비트 개수
            n_layers: 레이어 개수
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # SEL: n_layers × n_qubits × 3
        self.n_params = n_layers * n_qubits * 3
        
        print(f"✅ StronglyEntanglingLayers Ansatz initialized")
        print(f"   Qubits: {n_qubits}")
        print(f"   Layers: {n_layers}")
        print(f"   Parameters: {self.n_params}")
    
    def apply(self, params):
        """
        StronglyEntanglingLayers 적용
        
        Args:
            params: 파라미터 (n_params,) 또는 (n_layers, n_qubits, 3)
        """
        # Reshape to (n_layers, n_qubits, 3)
        if len(params.shape) == 1:
            weights = params.reshape(self.n_layers, self.n_qubits, 3)
        else:
            weights = params
        
        # PennyLane StronglyEntanglingLayers
        qml.StronglyEntanglingLayers(
            weights,
            wires=range(self.n_qubits),
            imprimitive=qml.CNOT  # CNOT 기반 얽힘
        )
    
    def get_num_params(self):
        """파라미터 개수 반환"""
        return self.n_params
