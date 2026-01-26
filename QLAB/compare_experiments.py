"""
여러 하이퍼파라미터 조합을 자동으로 실험하고 비교
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import VQCClassifier
from src.data import CompetitionDataLoader
from src.training import train_model

# 실험 설정들
EXPERIMENTS = [
    {
        'name': 'baseline',
        'n_layers': 5,
        'topology': 'linear',
        'measurement': [6, 7],
        'init_scale': 0.01,
        'lr': 0.05,
        'epochs': 200
    },
    {
        'name': 'layer_6',
        'n_layers': 6,
        'topology': 'linear',
        'measurement': [6, 7],
        'init_scale': 0.01,
        'lr': 0.05,
        'epochs': 200
    },
    {
        'name': 'init_scale_0.05',
        'n_layers': 5,
        'topology': 'linear',
        'measurement': [6, 7],
        'init_scale': 0.05,
        'lr': 0.05,
        'epochs': 200
    },
    {
        'name': 'layer_6_init_0.05',
        'n_layers': 6,
        'topology': 'linear',
        'measurement': [6, 7],
        'init_scale': 0.05,
        'lr': 0.05,
        'epochs': 200
    },
    {
        'name': 'layer_7',
        'n_layers': 7,
        'topology': 'linear',
        'measurement': [6, 7],
        'init_scale': 0.01,
        'lr': 0.05,
        'epochs': 200
    },
    {
        'name': 'epochs_300',
        'n_layers': 5,
        'topology': 'linear',
        'measurement': [6, 7],
        'init_scale': 0.01,
        'lr': 0.05,
        'epochs': 300
    },
]

def run_experiment(exp_config, train_loader, t_train_X, t_train_Y):
    """단일 실험 실행"""
    
    print(f"\n{'='*70}")
    print(f"  실험: {exp_config['name']}")
    print(f"{'='*70}")
    print(f"  레이어: {exp_config['n_layers']}")
    print(f"  토폴로지: {exp_config['topology']}")
    print(f"  측정 큐비트: {exp_config['measurement']}")
    print(f"  초기화: {exp_config['init_scale']}")
    print(f"  학습률: {exp_config['lr']}")
    print(f"  에폭: {exp_config['epochs']}")
    
    # 모델 생성
    model = VQCClassifier(
        n_qubits=8,
        n_layers=exp_config['n_layers'],
        topology=exp_config['topology'],
        measurement_qubits=exp_config['measurement'],
        init_scale=exp_config['init_scale']
    )
    
    # 훈련
    loss_history, acc_history, _ = train_model(
        model=model,
        train_loader=train_loader,
        epochs=exp_config['epochs'],
        lr=exp_config['lr'],
        loss_name='ce',
        optimizer_name='adam',
        device='cpu',
        verbose=False  # 간략하게
    )
    
    # 최종 평가
    model.eval()
    with torch.no_grad():
        predictions = model(t_train_X)
        pred_labels = torch.argmax(predictions, dim=1).numpy()
    
    true_labels = t_train_Y.numpy()
    accuracy = (true_labels == pred_labels).mean()
    
    result = {
        'name': exp_config['name'],
        'config': exp_config,
        'final_loss': loss_history[-1],
        'final_accuracy': accuracy,
        'best_accuracy': max(acc_history),
        'loss_history': loss_history,
        'acc_history': acc_history
    }
    
    print(f"\n  ✅ 완료!")
    print(f"  최종 정확도: {accuracy*100:.2f}%")
    print(f"  최고 정확도: {max(acc_history)*100:.2f}%")
    print(f"  최종 Loss: {loss_history[-1]:.4f}")
    
    return result


def main():
    """모든 실험 실행 및 비교"""
    
    print(f"\n{'='*70}")
    print("  하이퍼파라미터 비교 실험")
    print(f"{'='*70}")
    print(f"  총 {len(EXPERIMENTS)}개 실험 진행")
    
    # 데이터 로드
    data_loader = CompetitionDataLoader(data_dir='./', batch_size=4, shuffle=True)
    train_X, train_Y = data_loader.load_data()
    train_loader = data_loader.get_dataloader()
    t_train_X, t_train_Y = data_loader.get_torch_tensors()
    
    # 모든 실험 실행
    results = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp, train_loader, t_train_X, t_train_Y)
        results.append(result)
    
    # 결과 비교
    print(f"\n{'='*70}")
    print("  📊 전체 결과 비교")
    print(f"{'='*70}\n")
    
    # 정확도 기준 정렬
    results_sorted = sorted(results, key=lambda x: x['final_accuracy'], reverse=True)
    
    print(f"{'순위':>4} {'실험명':^25} {'최종 정확도':>12} {'최고 정확도':>12} {'최종 Loss':>12}")
    print("-" * 70)
    
    for rank, result in enumerate(results_sorted, 1):
        name = result['name']
        final_acc = result['final_accuracy'] * 100
        best_acc = result['best_accuracy'] * 100
        final_loss = result['final_loss']
        
        marker = "⭐" if rank == 1 else "  "
        print(f"{rank:4d} {marker} {name:23s} {final_acc:11.2f}% {best_acc:11.2f}% {final_loss:12.4f}")
    
    # 최고 설정 출력
    best = results_sorted[0]
    print(f"\n{'='*70}")
    print("  🏆 최고 성능 설정")
    print(f"{'='*70}")
    print(f"  실험명: {best['name']}")
    print(f"  최종 정확도: {best['final_accuracy']*100:.2f}%")
    print(f"\n설정:")
    for key, value in best['config'].items():
        if key != 'name':
            print(f"  {key}: {value}")
    
    print(f"\n{'='*70}")
    print("  💡 추천")
    print(f"{'='*70}")
    print(f"  '{best['name']}' 설정으로 최종 제출하는 것을 추천합니다!")
    print(f"{'='*70}\n")
    
    return results_sorted


if __name__ == "__main__":
    results = main()
