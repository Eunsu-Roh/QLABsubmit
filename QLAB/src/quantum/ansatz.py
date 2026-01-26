"""
Variational ansatz (parameterized quantum circuit)
"""

import pennylane as qml
import torch


class VariationalAnsatz:
    """
    학습 가능한 변분 회로
    
    구조:
    - Rotation layer (Rot gates)
    - Entangling layer (CNOT gates)
    """
    
    def __init__(self, n_qubits: int, n_layers: int, topology: str = 'linear'):
        """
        Args:
            n_qubits: 큐비트 개수
            n_layers: 레이어 개수
            topology: 얽힘 토폴로지 ('linear', 'ring', 'bidirectional')
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.topology = topology
        self.n_params = n_layers * 3 * n_qubits
        
        # 토폴로지 검증
        valid_topologies = ['linear', 'ring', 'bidirectional', 'all']
        if topology not in valid_topologies:
            raise ValueError(f"Invalid topology. Choose from {valid_topologies}")
    
    def apply(self, params):
        """
        Ansatz 적용
        
        Args:
            params: 파라미터 텐서 (n_layers * 3 * n_qubits,)
        """
        params = params.reshape(self.n_layers, 3, self.n_qubits)
        
        # 레이어 0 ~ n-2: Rotation + Entanglement
        for layer in range(self.n_layers - 1):
            self._rotation_layer(params[layer])
            self._entangling_layer()
        
        # 마지막 레이어: Rotation only
        self._rotation_layer(params[-1])
    
    def _rotation_layer(self, layer_params):
        """
        Rotation gates 레이어
        
        Args:
            layer_params: (3, n_qubits)
        """
        for i in range(self.n_qubits):
            qml.Rot(
                layer_params[0, i],
                layer_params[1, i],
                layer_params[2, i],
                wires=i
            )
    
    def _entangling_layer(self):
        """토폴로지에 따른 CNOT 레이어"""
        
        if self.topology == 'linear':
            # 선형: 0→1, 1→2, ..., 6→7
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        
        elif self.topology == 'ring':
            # 링: 선형 + 7→0
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits-1, 0])
        
        elif self.topology == 'bidirectional':
            # 양방향
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            for i in range(self.n_qubits - 1, 0, -1):
                qml.CNOT(wires=[i, i-1])
        
        elif self.topology == 'all':
            # 모든 qubit 쌍 간 연결
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
    
    def __repr__(self):
        return (f"VariationalAnsatz(n_qubits={self.n_qubits}, "
                f"n_layers={self.n_layers}, topology='{self.topology}')")
