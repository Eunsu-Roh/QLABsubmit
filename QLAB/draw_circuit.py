"""
양자 회로 시각화
"""

import os
import sys
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 더미 입력 (256차원 정규화된 벡터)
dummy_state = np.zeros(256, dtype=complex)
dummy_state[0] = 1.0  # |00000000⟩ 상태

# 더미 파라미터
n_layers = 5
n_qubits = 8
dummy_weights = np.random.randn(n_layers, n_qubits, 3) * 0.01

# 디바이스 생성
dev = qml.device('default.qubit', wires=8)

@qml.qnode(dev)
def circuit(inputs, weights):
    """baseline 회로"""
    
    # Feature Map: State Preparation
    qml.StatePrep(inputs, wires=range(8))
    
    # Variational Ansatz - 5 layers
    for layer in range(5):
        # Rotation layer
        for q in range(8):
            qml.Rot(weights[layer, q, 0], 
                   weights[layer, q, 1], 
                   weights[layer, q, 2], 
                   wires=q)
        
        # Entangling layer (linear topology)
        for q in range(7):
            qml.CNOT(wires=[q, q+1])
    
    # Measurement
    return qml.expval(qml.PauliZ(6)), qml.expval(qml.PauliZ(7))


# 회로 실행 (drawing을 위해)
_ = circuit(dummy_state, dummy_weights)

# 회로 그리기
print("\n회로 다이어그램 생성 중...\n")

fig, ax = qml.draw_mpl(circuit, style='black_white', show_all_wires=True)(dummy_state, dummy_weights)
plt.tight_layout()

# 저장
output_path = './outputs/circuit_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 회로 다이어그램 저장: {output_path}")

# 더 간단한 버전 (1 layer만)
@qml.qnode(dev)
def simple_circuit(inputs, weights):
    """1 레이어만 표시 (간단 버전)"""
    
    # Feature Map
    qml.StatePrep(inputs, wires=range(8))
    
    # 1 layer만
    for q in range(8):
        qml.Rot(weights[0, q, 0], 
               weights[0, q, 1], 
               weights[0, q, 2], 
               wires=q)
    
    for q in range(7):
        qml.CNOT(wires=[q, q+1])
    
    # Measurement
    return qml.expval(qml.PauliZ(6)), qml.expval(qml.PauliZ(7))

_ = simple_circuit(dummy_state, dummy_weights)

fig2, ax2 = qml.draw_mpl(simple_circuit, style='black_white', show_all_wires=True)(dummy_state, dummy_weights)
plt.tight_layout()

simple_path = './outputs/circuit_simple.png'
plt.savefig(simple_path, dpi=300, bbox_inches='tight')
print(f"✅ 단순 회로 다이어그램 저장: {simple_path}")

print("\n📊 회로 정보:")
print(f"   총 큐비트: 8")
print(f"   레이어 수: 5")
print(f"   파라미터: 120개")
print(f"   CNOT 게이트: 35개 (7 × 5 layers)")
print(f"   측정: 큐비트 [6, 7]\n")
