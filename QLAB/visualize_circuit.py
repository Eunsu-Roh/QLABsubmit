"""
QCNN 회로 구조 시각화 스크립트
"""
import os
import sys
import matplotlib.pyplot as plt
import pennylane as qml
import torch

# 프로젝트 루트 경로 설정 (src 모듈 import를 위해)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import VQCClassifier

def main():
    # 설정 (main.py의 Config와 동일하게 설정)
    config = {
        'n_qubits': 8,
        'n_layers': 5,
        'topology': 'linear',
        'measurement_qubits': [3, 4],
        'ansatz_type': 'variational',
        'measurement_type': 'computational'
    }

    print(f"🎨 회로 시각화 설정: {config}")

    # 모델 인스턴스 생성
    model = VQCClassifier(
        n_qubits=config['n_qubits'],
        n_layers=config['n_layers'],
        topology=config['topology'],
        measurement_qubits=config['measurement_qubits'],
        ansatz_type=config['ansatz_type'],
        measurement_type=config['measurement_type']
    )

    # Ansatz 회로 함수 가져오기
    qc = model.get_quantum_circuit()
    ansatz_circuit = qc.get_ansatz_circuit()
    params = model.params.detach()

    # 시각화용 전체 회로 구성 (중첩 + Ansatz + 측정)
    dev = qml.device('default.qubit', wires=config['n_qubits'])

    @qml.qnode(dev)
    def full_circuit(params):
        # 1. 중첩 (Superposition) - 모든 큐비트에 Hadamard 게이트
        for i in range(config['n_qubits']):
            qml.Hadamard(wires=i)
        
        # 2. Ansatz (학습된 회로)
        ansatz_circuit(params)
        
        # 3. 측정 (Measurement) - 지정된 큐비트 측정
        return [qml.expval(qml.PauliZ(i)) for i in config['measurement_qubits']]

    # 시각화 및 저장
    try:
        fig, ax = qml.draw_mpl(full_circuit, decimals=2)(params)
        fig.suptitle(f"Full Circuit ({config['ansatz_type']})", fontsize=16)
        
        save_path = os.path.join(project_root, 'circuit_diagram.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 회로도가 저장되었습니다: {save_path}")
        plt.show()
    except Exception as e:
        print(f"⚠️ 시각화 오류: {e}")
        print(qml.draw(ansatz_circuit)(params))

if __name__ == "__main__":
    main()