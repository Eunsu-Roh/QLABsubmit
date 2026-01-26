"""
Hardware Efficient Ansatz (HEA)

실제 양자 하드웨어에 최적화된 ansatz 구조
RY-RZ rotation + CZ entanglement 패턴
"""

import pennylane as qml


class HardwareEfficientAnsatz:
    """
    Hardware Efficient Ansatz
    
    구조 (per layer):
    1. RY rotation on all qubits
    2. RZ rotation on all qubits  
    3. CZ entanglement (linear or brick-wall)
    
    장점:
    - 실제 양자 컴퓨터에서 높은 fidelity
    - RY-RZ가 Bloch sphere 전체 커버
    - CZ gate는 CNOT보다 구현 쉬움
    """
    
    def __init__(self, n_qubits: int, n_layers: int, entanglement: str = 'linear'):
        """
        Args:
            n_qubits: 큐비트 개수
            n_layers: 레이어 개수
            entanglement: 'linear', 'circular', 'brick_wall'
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        
        # 파라미터 개수: 각 layer마다 RY(n) + RZ(n) = 2n
        self.n_params = 2 * n_qubits * n_layers
    
    def _rotation_layer(self, params_ry, params_rz):
        """
        RY + RZ rotation layer
        
        Args:
            params_ry: RY angles (n_qubits,)
            params_rz: RZ angles (n_qubits,)
        """
        for i in range(self.n_qubits):
            qml.RY(params_ry[i], wires=i)
            qml.RZ(params_rz[i], wires=i)
    
    def _entangling_layer_linear(self):
        """Linear entanglement: CZ on adjacent qubits"""
        for i in range(self.n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
    
    def _entangling_layer_linear_cnot(self):
        """Linear entanglement: CNOT on adjacent qubits"""
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    def _entangling_layer_circular(self):
        """Circular entanglement: linear + wrap around"""
        self._entangling_layer_linear()
        qml.CZ(wires=[self.n_qubits - 1, 0])
    
    def _entangling_layer_brick_wall(self, layer_idx):
        """
        Brick-wall pattern:
        - Even layers: (0,1), (2,3), (4,5), (6,7)
        - Odd layers:  (1,2), (3,4), (5,6), (7,0)
        """
        if layer_idx % 2 == 0:
            # Even layer: pair (0,1), (2,3), ...
            for i in range(0, self.n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
        else:
            # Odd layer: pair (1,2), (3,4), ...
            for i in range(1, self.n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
            if self.n_qubits > 2:
                qml.CZ(wires=[self.n_qubits - 1, 0])  # Wrap
    
    def apply(self, params):
        """
        Apply Hardware Efficient Ansatz
        
        Args:
            params: flattened parameters (2 * n_qubits * n_layers,)
        """
        # Reshape: (n_layers, 2, n_qubits)
        params = params.reshape(self.n_layers, 2, self.n_qubits)
        
        for layer in range(self.n_layers):
            # Rotation layer
            params_ry = params[layer, 0, :]
            params_rz = params[layer, 1, :]
            self._rotation_layer(params_ry, params_rz)
            
            # Entanglement layer
            if self.entanglement == 'linear':
                self._entangling_layer_linear()
            elif self.entanglement == 'linear_cnot':
                self._entangling_layer_linear_cnot()
            elif self.entanglement == 'circular':
                self._entangling_layer_circular()
            elif self.entanglement == 'brick_wall':
                self._entangling_layer_brick_wall(layer)
            else:
                raise ValueError(f"Unknown entanglement: {self.entanglement}")
    
    def __repr__(self):
        return (f"HardwareEfficientAnsatz(n_qubits={self.n_qubits}, "
                f"n_layers={self.n_layers}, entanglement='{self.entanglement}')")


class AlternatingLayeredAnsatz:
    """
    Alternating Layered Ansatz
    
    구조:
    1. RY layer (all qubits)
    2. Entanglement layer
    3. RZ layer (all qubits)
    4. Entanglement layer
    
    특징:
    - Rotation과 entanglement 완전 분리
    - 더 깊은 표현력
    """
    
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # RY layer + RZ layer per ansatz layer = 2n params/layer
        self.n_params = 2 * n_qubits * n_layers
    
    def apply(self, params):
        """Apply alternating ansatz"""
        params = params.reshape(self.n_layers, 2, self.n_qubits)
        
        for layer in range(self.n_layers):
            # RY layer
            for i in range(self.n_qubits):
                qml.RY(params[layer, 0, i], wires=i)
            
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # RZ layer
            for i in range(self.n_qubits):
                qml.RZ(params[layer, 1, i], wires=i)
            
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])


class SimplifiedTwoLocalAnsatz:
    """
    Simplified Two-Local Ansatz
    
    Qiskit의 TwoLocal과 유사하지만 단순화
    - Single qubit: RY만 사용
    - Two qubit: CX만 사용
    - Full entanglement 옵션
    """
    
    def __init__(self, n_qubits: int, n_layers: int, entanglement: str = 'full'):
        """
        Args:
            entanglement: 'linear', 'full', 'circular'
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        
        # RY on all qubits per layer
        self.n_params = n_qubits * n_layers
    
    def apply(self, params):
        """Apply two-local ansatz"""
        params = params.reshape(self.n_layers, self.n_qubits)
        
        for layer in range(self.n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RY(params[layer, i], wires=i)
            
            # Entanglement layer
            if self.entanglement == 'linear':
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            elif self.entanglement == 'full':
                # Full entanglement: all pairs
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[i, j])
            elif self.entanglement == 'circular':
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
