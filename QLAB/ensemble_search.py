"""
앙상블 접근: 여러 Random Seed로 학습하여 최적 모델 탐색

다양한 초기화로 local optima를 탐색하여 더 나은 성능 발견
"""

import os
import sys
import json
import torch
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import VQCClassifier
from src.data import CompetitionDataLoader
from src.training import train_model
from src.utils import evaluate_model


def train_with_seed(seed, config):
    """특정 seed로 모델 훈련"""
    print(f"\n{'='*70}")
    print(f"  Seed {seed} 훈련")
    print(f"{'='*70}")
    
    # Seed 설정
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 데이터 로드
    data_loader = CompetitionDataLoader(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    train_X, train_Y = data_loader.load_data()
    train_loader = data_loader.get_dataloader()
    t_train_X, t_train_Y = data_loader.get_torch_tensors()
    
    # 모델 생성
    model = VQCClassifier(
        n_qubits=config['n_qubits'],
        n_layers=config['n_layers'],
        topology=config['topology'],
        measurement_qubits=config['measurement_qubits'],
        ansatz_type=config['ansatz_type'],
        init_scale=config['init_scale']
    )
    
    # 훈련
    loss_history, acc_history, _ = train_model(
        model,
        train_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        loss_name=config['loss_function'],
        verbose=True
    )
    
    # 평가
    pred_labels, pred_probs, accuracy = evaluate_model(
        model, t_train_X, t_train_Y, verbose=False
    )
    
    return {
        'seed': seed,
        'model': model,
        'accuracy': accuracy,
        'loss': loss_history[-1],
        'pred_labels': pred_labels,
        'pred_probs': pred_probs
    }


def main():
    """여러 seed로 실험"""
    
    # 실험 설정
    config = {
        'n_qubits': 8,
        'n_layers': 5,
        'topology': 'linear',
        'measurement_qubits': [3, 4],
        'ansatz_type': 'variational',
        'init_scale': 0.01,
        'batch_size': 4,
        'learning_rate': 0.05,
        'epochs': 200,
        'loss_function': 'ce',
        'data_dir': './'
    }
    
    # 시도할 seed 리스트 (빠른 테스트용: 3개만)
    seeds = [42, 123, 789]
    
    print(f"\n{'='*70}")
    print(f"  앙상블 탐색: {len(seeds)}개 Random Seeds")
    print(f"{'='*70}")
    print(f"\nSeeds: {seeds}")
    
    results = []
    
    # 각 seed로 훈련
    for seed in seeds:
        try:
            result = train_with_seed(seed, config)
            results.append(result)
            
            print(f"\n✅ Seed {seed}: Accuracy = {result['accuracy']:.2%}, Loss = {result['loss']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Seed {seed} 실패: {e}")
    
    # 결과 정렬 (정확도 높은 순)
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n\n{'='*70}")
    print(f"  결과 요약 (정확도 순)")
    print(f"{'='*70}")
    print(f"\n{'Rank':<6} {'Seed':<8} {'Accuracy':<12} {'Loss':<10}")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<6} {result['seed']:<8} {result['accuracy']:<12.2%} {result['loss']:<10.4f}")
    
    # 최고 성능 모델 저장
    best_result = results[0]
    
    print(f"\n{'='*70}")
    print(f"  최고 성능 모델")
    print(f"{'='*70}")
    print(f"Seed: {best_result['seed']}")
    print(f"Accuracy: {best_result['accuracy']:.2%}")
    print(f"Loss: {best_result['loss']:.4f}")
    
    # Submission 생성
    print(f"\n{'='*70}")
    print(f"  제출 파일 생성 (Seed {best_result['seed']})")
    print(f"{'='*70}")
    
    import pennylane as qml
    
    model = best_result['model']
    qc = model.get_quantum_circuit()
    trained_params = model.params.detach().cpu()
    ansatz_circuit = qc.get_ansatz_circuit()
    
    qasm_string = qml.to_openqasm(ansatz_circuit, measure_all=False)(trained_params)
    
    submission = {
        "measurements": qc.measurement_qubits,
        "qasm": qasm_string
    }
    
    output_path = f'./outputs/submission_seed{best_result["seed"]}.json'
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    # 기본 submission도 덮어쓰기
    with open('./outputs/submission.json', 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"✅ 저장 완료: {output_path}")
    print(f"✅ 기본 파일 업데이트: ./outputs/submission.json")
    
    # 결과 로그 저장
    log_data = {
        'best_seed': best_result['seed'],
        'best_accuracy': float(best_result['accuracy']),
        'best_loss': float(best_result['loss']),
        'all_results': [
            {
                'seed': r['seed'],
                'accuracy': float(r['accuracy']),
                'loss': float(r['loss'])
            }
            for r in results
        ]
    }
    
    with open('./outputs/ensemble_results.json', 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"✅ 결과 로그: ./outputs/ensemble_results.json")


if __name__ == '__main__':
    main()
