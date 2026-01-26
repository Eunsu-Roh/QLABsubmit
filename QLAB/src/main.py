"""
2nd Quantum AI Competition 2026
Main execution script
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import VQCClassifier
from src.data import CompetitionDataLoader
from src.training import train_model
from src.utils import evaluate_model, plot_training_history, plot_confusion_matrix


class Config:
    """실험 설정"""
    
    # 회로 구조
    N_QUBITS = 8
    N_LAYERS = 5  # 93.3% 모델 설정 (표현력 확보)
    TOPOLOGY = 'linear'
    MEASUREMENT_QUBITS = [3, 4]  # 93.3% 모델 설정 (중앙 큐비트가 정보 전파 수신에 유리)
    MEASUREMENT_TYPE = 'computational'  # 최종 제출용: Z기저 측정 및 QASM 생성
    
    # Ansatz 종류:
    ANSATZ_TYPE = 'variational'  # 검증된 VQC 구조
    
    # 훈련
    BATCH_SIZE = 4
    LEARNING_RATE = 0.035
    EPOCHS = 260  # VQC는 수렴이 빠르므로 300이면 충분
    LOSS_FUNCTION = 'focal'  # Hard Sample(Label 3) 집중 학습을 위해 Focal Loss 사용
    OPTIMIZER = 'adam'
    WEIGHT_DECAY = 0.0001  # L2 규제 (Smoothing과 함께 사용 시 시너지 효과)
    SMOOTHING = 0.0        # 오분류가 심하므로 Smoothing 제거하여 결정 경계 명확화
    SCHEDULER = None
    FOCAL_GAMMA = 3.0   # 강도 증가 (Label 3 오분류 해결)
    
    # Class Balancing: Focal Loss가 자동으로 어려운 샘플에 가중치를 부여하므로 수동 설정 제거
    USE_CLASS_WEIGHTS = False
    CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
    
    # 경로
    DATA_DIR = './'
    OUTPUT_DIR = './outputs'
    
    # 재현성
    RANDOM_SEED = 42
    
    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        torch.manual_seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)


def create_submission(model, output_dir='./outputs'):
    """
    제출 파일 생성
    
    Args:
        model: VQCClassifier
        output_dir: 출력 디렉토리
    
    Returns:
        submission dict
    """
    import pennylane as qml
    import json
    
    print(f"\n{'='*70}")
    print("  제출 파일 생성")
    print(f"{'='*70}")
    
    # 양자 회로 가져오기
    qc = model.get_quantum_circuit()
    trained_params = model.params.detach().cpu()
    
    # Ansatz만 있는 회로
    ansatz_circuit = qc.get_ansatz_circuit()
    
    # QASM 생성 (QASM 2.0 형식 - 서버 호환)
    try:
        # PennyLane의 기본 QASM 2.0 생성
        qasm_string = qml.to_openqasm(ansatz_circuit, measure_all=False)(trained_params)
        
        print(f"📝 QASM 형식: OPENQASM 2.0 (서버 호환)")
        
    except Exception as e:
        print(f"⚠️  QASM 생성 오류: {e}")
        print(f"   대체 방법 시도 중...")
        qasm_string = str(qml.draw(ansatz_circuit)(trained_params))
        raise e
    
    # Submission 구조
    submission = {
        "measurements": qc.measurement_qubits,
        "qasm": qasm_string
    }
    
    # JSON 저장
    output_path = os.path.join(output_dir, 'submission.json')
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"✅ 제출 파일 생성 완료!")
    print(f"   파일: {output_path}")
    print(f"   측정 큐비트: {qc.measurement_qubits}")
    print(f"   QASM 길이: {len(qasm_string)} chars")
    
    if qc.measurement_type == 'pauli':
        print(f"\n⚠️  [주의] Pauli 측정 모드 감지됨")
        print(f"   제출용 QASM에는 Pauli->Probs 선형 레이어가 포함되지 않습니다.")
        print(f"   서버는 Z기저(Computational)로만 측정하므로 성능이 저하될 수 있습니다.")
        print(f"   전략: 이 모델로 위상 특징을 분석한 후, 'computational' 모드로 재학습하세요.")
    
    # 게이트 통계
    gate_counts = {
        'RZ': qasm_string.count('rz('),
        'RY': qasm_string.count('ry('),
        'CX': qasm_string.count('cx ')
    }
    print(f"\n🔧 게이트 통계:")
    for gate, count in gate_counts.items():
        print(f"   {gate}: {count}")
    
    return submission


def save_experiment_log(config, accuracy, loss_history, output_dir='./outputs'):
    """실험 로그 저장"""
    import json
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        "timestamp": timestamp,
        "config": {
            "n_qubits": config.N_QUBITS,
            "n_layers": config.N_LAYERS,
            "topology": config.TOPOLOGY,
            "measurement_qubits": config.MEASUREMENT_QUBITS,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "loss_function": config.LOSS_FUNCTION,
        },
        "results": {
            "final_accuracy": float(accuracy),
            "final_loss": float(loss_history[-1]),
            "best_loss": float(min(loss_history)),
        }
    }
    
    log_file = os.path.join(output_dir, 'experiments.log')
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry, indent=2))
        f.write("\n" + "="*70 + "\n\n")
    
    print(f"\n💾 실험 로그 저장: {log_file}")


def main():
    """메인 파이프라인"""
    
    print(f"\n{'='*70}")
    print("  2nd Quantum AI Competition 2026")
    print("  Quantum Phase Classification")
    print(f"{'='*70}")
    
    # 설정
    config = Config()
    
    print(f"\n🎯 실험 설정:")
    print(f"   큐비트: {config.N_QUBITS}")
    print(f"   레이어: {config.N_LAYERS}")
    print(f"   Ansatz: {config.ANSATZ_TYPE}")
    print(f"   토폴로지: {config.TOPOLOGY}")
    print(f"   측정 큐비트: {config.MEASUREMENT_QUBITS}")
    print(f"   측정 방식: {config.MEASUREMENT_TYPE}")
    print(f"   학습률: {config.LEARNING_RATE}")
    print(f"   에폭: {config.EPOCHS}")
    
    # ===== 1. 데이터 로드 =====
    data_loader = CompetitionDataLoader(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    train_X, train_Y = data_loader.load_data()
    
    train_loader = data_loader.get_dataloader()
    t_train_X, t_train_Y = data_loader.get_torch_tensors()
    
    # ===== 2. 모델 생성 =====
    model = VQCClassifier(
        n_qubits=config.N_QUBITS,
        n_layers=config.N_LAYERS,
        topology=config.TOPOLOGY,
        measurement_qubits=config.MEASUREMENT_QUBITS,
        ansatz_type=config.ANSATZ_TYPE,
        measurement_type=config.MEASUREMENT_TYPE,
        init_scale=0.01
    )
    
    device = 'cpu'
    
    # Class weights preparation
    class_weights = None
    if getattr(config, 'USE_CLASS_WEIGHTS', False):
        class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)

    # ===== 3. 훈련 =====
    loss_history, acc_history, best_params = train_model(
        model=model,
        train_loader=train_loader,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        loss_name=config.LOSS_FUNCTION,
        optimizer_name=config.OPTIMIZER,
        device=device,
        verbose=True,
        scheduler_name=getattr(config, 'SCHEDULER', None),
        weight_decay=getattr(config, 'WEIGHT_DECAY', 0.0),
        smoothing=getattr(config, 'SMOOTHING', 0.0),
        class_weights=class_weights,
        gamma=getattr(config, 'FOCAL_GAMMA', 2.0)
    )
    
    # ===== 4. 평가 =====
    pred_labels, pred_probs, accuracy = evaluate_model(
        model, t_train_X, t_train_Y, device, verbose=True
    )
    
    # ===== 5. 시각화 =====
    plot_training_history(
        loss_history, acc_history,
        save_path=os.path.join(config.OUTPUT_DIR, 'training_history.png')
    )
    
    plot_confusion_matrix(
        train_Y, pred_labels,
        save_path=os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
    )
    
    # ===== 5.1 검증 데이터 평가 (Verification) =====
    print(f"\n{'='*70}")
    print("  심사위원 검증 데이터 평가 (test_Xthis, test_ythis)")
    print(f"{'='*70}")
    
    test_X_path = os.path.join(config.DATA_DIR, 'test_Xthis.npy')
    test_Y_path = os.path.join(config.DATA_DIR, 'test_ythis.npy')
    
    if os.path.exists(test_X_path) and os.path.exists(test_Y_path):
        try:
            print(f"[INFO] 검증 데이터 로드 중...")
            test_X = np.load(test_X_path)
            test_Y = np.load(test_Y_path)
            print(f"[INFO] 데이터 크기: {test_X.shape}")
            
            # Tensor 변환
            t_test_X = torch.tensor(test_X, dtype=torch.complex64)
            t_test_Y = torch.tensor(test_Y, dtype=torch.long)
            
            # 평가
            print(f"[INFO] 2000개 데이터에 대한 추론 시작...")
            test_pred_labels, test_pred_probs, test_accuracy = evaluate_model(
                model, t_test_X, t_test_Y, device, verbose=True
            )
            
            print(f"\n🏆 검증 결과 (Verification Result):")
            print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
            
            # Confusion Matrix for test set
            plot_confusion_matrix(
                test_Y, test_pred_labels,
                save_path=os.path.join(config.OUTPUT_DIR, 'confusion_matrix_test.png')
            )
            
            # 상세 로그 저장 (verification_results.txt)
            log_path = os.path.join(config.OUTPUT_DIR, 'verification_results.txt')
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"Verification Results (test_Xthis, test_ythis)\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Accuracy: {test_accuracy*100:.2f}%\n")
                f.write(f"{'-'*55}\n")
                f.write(f"Index | True | Pred | Correct | Confidence\n")
                f.write(f"{'-'*55}\n")
                for i in range(len(test_Y)):
                    is_correct = "O" if test_Y[i] == test_pred_labels[i] else "X"
                    conf = float(test_pred_probs[i][test_pred_labels[i]])
                    f.write(f"{i:5d} | {test_Y[i]:4d} | {test_pred_labels[i]:4d} | {is_correct:7s} | {conf:.4f}\n")
            print(f"📝 상세 추론 로그 저장됨: {log_path}")
            
        except Exception as e:
            print(f"❌ 검증 중 오류 발생: {e}")
    else:
        print("⚠️ 검증 데이터 파일이 존재하지 않습니다. (test_Xthis.npy, test_ythis.npy)")
        print("   기존 학습 데이터(16개)에 대한 결과만 출력되었습니다.")

    # ===== 5.5 Pauli 가중치 분석 (Pauli 모드일 경우) =====
    if config.MEASUREMENT_TYPE == 'pauli':
        print(f"\n🔍 Pauli Operator Weights Analysis (Top 3 per Class):")
        weights = model.pauli_to_probs.weight.detach().cpu().numpy()
        # weights shape: (4, 10) -> (Classes, Observables)
        observables = ['ZZ', 'XX', 'YY', 'Z0', 'Z1', 'I', 'X0', 'X1', 'Y0', 'Y1']
        classes = ['Cluster', 'Trivial', 'Ferro', 'Anti-Ferro']
        
        for i, cls in enumerate(classes):
            print(f"   Class {cls}:")
            # 절대값 기준 정렬
            w = weights[i]
            indices = np.argsort(-np.abs(w))
            for idx in indices[:3]: # Top 3
                print(f"     {observables[idx]:<3}: {w[idx]:.4f}")

    # ===== 6. 제출 파일 생성 =====
    submission = create_submission(model, config.OUTPUT_DIR)
    
    # ===== 7. 실험 로그 =====
    save_experiment_log(config, accuracy, loss_history, config.OUTPUT_DIR)
    
    # ===== 최종 요약 =====
    print(f"\n{'='*70}")
    print("  🎉 모든 작업 완료!")
    print(f"{'='*70}")
    print(f"\n📈 최종 결과:")
    print(f"   훈련 정확도: {accuracy*100:.2f}%")
    print(f"   최종 손실: {loss_history[-1]:.4f}")
    print(f"\n📁 생성된 파일:")
    print(f"   - {os.path.join(config.OUTPUT_DIR, 'submission.json')}")
    print(f"   - {os.path.join(config.OUTPUT_DIR, 'training_history.png')}")
    print(f"   - {os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')}")
    print(f"   - {os.path.join(config.OUTPUT_DIR, 'experiments.log')}")
    print(f"\n🚀 다음 단계:")
    print(f"   1. submission.json을 AI Factory에 제출")
    print(f"   2. Config 클래스 수정하여 다른 하이퍼파라미터 실험")
    print(f"   3. 리더보드에서 결과 확인")
    print(f"{'='*70}\n")
    
    return model, submission


if __name__ == "__main__":
    model, submission = main()
