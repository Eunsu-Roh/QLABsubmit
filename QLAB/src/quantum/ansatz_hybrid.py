"""
Hybrid Ansatz: HEA + Variational 조합

HEA의 Label 1,2 강점 + Variational의 Label 0,3 강점 결합
"""

import pennylane as qml


class HybridAnsatz:
    """
    Hybrid Ansatz: HEA layers + Variational layers
    
    구조:
    1. HEA layers (앞부분) - Label 1,2 특징 추출
    2. Variational layers (뒷부분) - Label 0,3 보완
    
    목표: 모든 클래스에 대해 균형잡힌 성능
    """
    
    def __init__(self, n_qubits: int, n_hea_layers: int, n_var_layers: int):
        """
        Args:
            n_qubits: 큐비트 개수
            n_hea_layers: HEA 레이어 개수
            n_var_layers: Variational 레이어 개수
        """
        self.n_qubits = n_qubits
        self.n_hea_layers = n_hea_layers
        self.n_var_layers = n_var_layers
        
        # HEA params: 2 * n_qubits * n_hea_layers (RY + RZ)
        # Variational params: 3 * n_qubits * n_var_layers (Rot)
        self.n_params = (2 * n_qubits * n_hea_layers) + (3 * n_qubits * n_var_layers)
    
    def _hea_layer(self, params_ry, params_rz):
        """HEA rotation layer"""
        for i in range(self.n_qubits):
            qml.RY(params_ry[i], wires=i)
            qml.RZ(params_rz[i], wires=i)
    
    def _hea_entangle(self):
        """HEA entanglement (linear)"""
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    def _variational_layer(self, params):
        """Variational rotation layer"""
        for i in range(self.n_qubits):
            qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=i)
    
    def _variational_entangle(self):
        """Variational entanglement (linear)"""
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    def apply(self, params):
        """
        Apply hybrid ansatz
        
        Args:
            params: flattened parameters
        """
        # Split params
        hea_size = 2 * self.n_qubits * self.n_hea_layers
        hea_params = params[:hea_size].reshape(self.n_hea_layers, 2, self.n_qubits)
        var_params = params[hea_size:].reshape(self.n_var_layers, self.n_qubits, 3)
        
        # 1. HEA layers (feature extraction for Label 1,2)
        for layer in range(self.n_hea_layers):
            self._hea_layer(hea_params[layer, 0, :], hea_params[layer, 1, :])
            self._hea_entangle()
        
        # 2. Variational layers (refinement for Label 0,3)
        for layer in range(self.n_var_layers):
            self._variational_layer(var_params[layer])
            self._variational_entangle()
    
    def __repr__(self):
        return (f"HybridAnsatz(n_qubits={self.n_qubits}, "
                f"hea_layers={self.n_hea_layers}, var_layers={self.n_var_layers})")


class AlternatingHybridAnsatz:
    """
    Alternating Hybrid: HEA와 Variational을 번갈아 사용
    
    HEA layer → Var layer → HEA layer → Var layer ...
    """
    
    def __init__(self, n_qubits: int, n_layers: int):
        """
        Args:
            n_qubits: 큐비트 개수
            n_layers: 전체 레이어 개수 (HEA와 Var을 번갈아)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 짝수 layer: HEA (2 params), 홀수 layer: Var (3 params)
        # 평균 2.5 params/qubit/layer
        hea_layers = n_layers // 2
        var_layers = n_layers - hea_layers
        self.n_params = (2 * n_qubits * hea_layers) + (3 * n_qubits * var_layers)
    
    def apply(self, params):
        """Apply alternating hybrid ansatz"""
        param_idx = 0
        
        for layer in range(self.n_layers):
            if layer % 2 == 0:
                # Even: HEA layer
                params_ry = params[param_idx:param_idx + self.n_qubits]
                param_idx += self.n_qubits
                params_rz = params[param_idx:param_idx + self.n_qubits]
                param_idx += self.n_qubits
                
                for i in range(self.n_qubits):
                    qml.RY(params_ry[i], wires=i)
                    qml.RZ(params_rz[i], wires=i)
            else:
                # Odd: Variational layer
                rot_params = params[param_idx:param_idx + 3 * self.n_qubits].reshape(self.n_qubits, 3)
                param_idx += 3 * self.n_qubits
                
                for i in range(self.n_qubits):
                    qml.Rot(rot_params[i, 0], rot_params[i, 1], rot_params[i, 2], wires=i)
            
            # Entanglement after each layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])


class BalancedHybridAnsatz:
    """
    Balanced Hybrid: 클래스 균형을 위한 설계
    
    - 첫 절반: HEA (Label 1,2)
    - 중간: Transition layer
    - 후반: Variational (Label 0,3)
    """
    
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 1/3 HEA, 1/3 transition, 1/3 Var
        self.hea_layers = n_layers // 3
        self.var_layers = n_layers // 3
        self.transition_layers = n_layers - self.hea_layers - self.var_layers
        
        hea_params = 2 * n_qubits * self.hea_layers
        var_params = 3 * n_qubits * self.var_layers
        trans_params = 3 * n_qubits * self.transition_layers  # Rot gates
        
        self.n_params = hea_params + var_params + trans_params
    
    def apply(self, params):
        """Apply balanced hybrid ansatz"""
        param_idx = 0
        
        # Phase 1: HEA
        for _ in range(self.hea_layers):
            params_ry = params[param_idx:param_idx + self.n_qubits]
            param_idx += self.n_qubits
            params_rz = params[param_idx:param_idx + self.n_qubits]
            param_idx += self.n_qubits
            
            for i in range(self.n_qubits):
                qml.RY(params_ry[i], wires=i)
                qml.RZ(params_rz[i], wires=i)
            
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Phase 2: Transition (Rot gates)
        for _ in range(self.transition_layers):
            rot_params = params[param_idx:param_idx + 3 * self.n_qubits].reshape(self.n_qubits, 3)
            param_idx += 3 * self.n_qubits
            
            for i in range(self.n_qubits):
                qml.Rot(rot_params[i, 0], rot_params[i, 1], rot_params[i, 2], wires=i)
            
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Phase 3: Variational
        for _ in range(self.var_layers):
            rot_params = params[param_idx:param_idx + 3 * self.n_qubits].reshape(self.n_qubits, 3)
            param_idx += 3 * self.n_qubits
            
            for i in range(self.n_qubits):
                qml.Rot(rot_params[i, 0], rot_params[i, 1], rot_params[i, 2], wires=i)
            
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
