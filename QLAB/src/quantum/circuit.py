"""
Complete quantum circuit combining feature map and ansatz
"""

import pennylane as qml
import torch
from .feature_map import StatePreparation
from .ansatz import VariationalAnsatz
from .ansatz_sel import StronglyEntanglingAnsatz
from .ansatz_hea import HardwareEfficientAnsatz, AlternatingLayeredAnsatz, SimplifiedTwoLocalAnsatz
from .ansatz_hybrid import HybridAnsatz, AlternatingHybridAnsatz, BalancedHybridAnsatz
from .ansatz_qcnn import QCNNAnsatz
from .ansatz_ising import IsingAnsatz


class QuantumCircuit:
    """
    완전한 양자 회로
    
    구성:
    1. State Preparation (feature map)
    2. Variational Ansatz (학습 가능)
    3. Measurement
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 5,
        topology: str = 'linear',
        measurement_qubits: list = [6, 7],
        device_name: str = 'default.qubit',
        ansatz_type: str = 'variational',  # 'variational' or 'sel'
        measurement_type: str = 'computational'  # 'computational' or 'pauli'
    ):
        """
        Args:
            n_qubits: 큐비트 개수
            n_layers: ansatz 레이어 개수
            topology: 얽힘 토폴로지 (variational ansatz용)
            measurement_qubits: 측정할 큐비트 (2개)
            device_name: PennyLane device
            ansatz_type: ansatz 종류 ('variational' or 'sel')
            measurement_type: 측정 방식 ('computational' or 'pauli')
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.topology = topology
        self.measurement_qubits = measurement_qubits
        self.ansatz_type = ansatz_type
        self.measurement_type = measurement_type
        
        # 측정 큐비트 검증
        if len(measurement_qubits) != 2:
            raise ValueError("Exactly 2 measurement qubits required")
        if not all(0 <= q < n_qubits for q in measurement_qubits):
            raise ValueError(f"Measurement qubits must be in range [0, {n_qubits-1}]")

        if ansatz_type == 'qcnn' and measurement_qubits != [5, 7]:
            print("[WARN] QCNN ansatz 권장 측정 큐비트는 [5, 7] 입니다.")
            print(f"   현재 설정: {measurement_qubits} (필요시 Config에서 변경 추천)")
        
        # Components
        self.feature_map = StatePreparation(n_qubits)
        
        # Ansatz 선택
        if ansatz_type == 'sel':
            self.ansatz = StronglyEntanglingAnsatz(n_qubits, n_layers)
        elif ansatz_type == 'qcnn':
            # QCNN-style hierarchical compression.
            # Recommended measurement qubits: [5, 7]
            self.ansatz = QCNNAnsatz(n_qubits, n_layers)
        elif ansatz_type == 'hea':
            self.ansatz = HardwareEfficientAnsatz(n_qubits, n_layers, topology)
        elif ansatz_type == 'alternating':
            self.ansatz = AlternatingLayeredAnsatz(n_qubits, n_layers)
        elif ansatz_type == 'two_local':
            self.ansatz = SimplifiedTwoLocalAnsatz(n_qubits, n_layers, topology)
        elif ansatz_type == 'hybrid':
            # HEA(2) + Var(3) = 5 layers total
            self.ansatz = HybridAnsatz(n_qubits, n_hea_layers=2, n_var_layers=3)
        elif ansatz_type == 'alt_hybrid':
            self.ansatz = AlternatingHybridAnsatz(n_qubits, n_layers)
        elif ansatz_type == 'balanced':
            self.ansatz = BalancedHybridAnsatz(n_qubits, n_layers)
        elif ansatz_type == 'ising':
            self.ansatz = IsingAnsatz(n_qubits, n_layers)
        else:
            self.ansatz = VariationalAnsatz(n_qubits, n_layers, topology)
        
        # PennyLane device
        self.dev = qml.device(device_name, wires=n_qubits)
        
        # Build circuits
        self._build_main_circuit()
        self._build_ansatz_only_circuit()
        
        print(f"✅ Quantum Circuit initialized")
        print(f"   Qubits: {n_qubits}")
        print(f"   Layers: {n_layers}")
        print(f"   Topology: {topology}")
        print(f"   Measurement: {measurement_qubits} ({measurement_type})")
        print(f"   Parameters: {self.ansatz.n_params}")
    
    def _build_main_circuit(self):
        """전체 회로 (훈련용)"""
        
        if self.measurement_type == 'pauli':
            # Pauli 측정: ZZ, XX, YY 각각의 expectation value
            @qml.qnode(self.dev, interface='torch')
            def circuit(state, params):
                """
                Args:
                    state: 양자 상태 (256,)
                    params: ansatz 파라미터
                
                Returns:
                    10개 값: [ZZ, XX, YY, Z0, Z1, I, X0, X1, Y0, Y1]
                    - ZZ, XX, YY: 2-qubit correlation
                    - Z0, Z1: single qubit expectations (Z basis)
                    - I: identity (normalization check)
                    - X0, X1, Y0, Y1: single qubit expectations (X, Y basis)
                """
                # 1. Feature map
                self.feature_map.apply(state, wires=range(self.n_qubits))
                
                # 2. Variational ansatz
                self.ansatz.apply(params)
                
                # 3. Pauli measurements
                q0, q1 = self.measurement_qubits
                
                zz = qml.expval(qml.PauliZ(q0) @ qml.PauliZ(q1))
                xx = qml.expval(qml.PauliX(q0) @ qml.PauliX(q1))
                yy = qml.expval(qml.PauliY(q0) @ qml.PauliY(q1))
                z0 = qml.expval(qml.PauliZ(q0))
                z1 = qml.expval(qml.PauliZ(q1))
                identity = qml.expval(qml.Identity(q0))
                
                # Add single qubit X and Y measurements (for Trivial/Cluster phases)
                x0 = qml.expval(qml.PauliX(q0))
                x1 = qml.expval(qml.PauliX(q1))
                y0 = qml.expval(qml.PauliY(q0))
                y1 = qml.expval(qml.PauliY(q1))
                
                return [zz, xx, yy, z0, z1, identity, x0, x1, y0, y1]
            
            self.circuit = circuit
            
        else:
            # Computational basis 측정 (기존)
            @qml.qnode(self.dev, interface='torch')
            def circuit(state, params):
                """
                Args:
                    state: 양자 상태 (256,)
                    params: ansatz 파라미터
                
                Returns:
                    4개 확률 (2-qubit 측정)
                """
                # 1. Feature map
                self.feature_map.apply(state, wires=range(self.n_qubits))
                
                # 2. Variational ansatz
                self.ansatz.apply(params)
                
                # 3. Measurement
                return qml.probs(wires=self.measurement_qubits)
            
            self.circuit = circuit
    
    def _build_ansatz_only_circuit(self):
        """Ansatz만 (QASM 출력용)"""
        
        @qml.qnode(self.dev, interface='torch')
        def ansatz_only(params):
            self.ansatz.apply(params)
            # No return - for QASM export
        
        self.ansatz_circuit = ansatz_only
    
    def forward(self, state, params):
        """
        Forward pass
        
        Args:
            state: 양자 상태
            params: ansatz 파라미터
        
        Returns:
            측정 확률
        """
        output = self.circuit(state, params)
        
        # Pauli 측정일 경우 tuple로 반환되므로 stack 필요
        if isinstance(output, (tuple, list)):
            return torch.stack(output, dim=-1).float()
            
        return output.float()
    
    def get_ansatz_circuit(self):
        """QASM 출력을 위한 ansatz 회로 반환"""
        return self.ansatz_circuit
    
    def __repr__(self):
        return (f"QuantumCircuit(n_qubits={self.n_qubits}, "
                f"n_layers={self.n_layers}, topology='{self.topology}')")
