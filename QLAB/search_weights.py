"""
Grid Search for Class Weights & Seeds
최적의 가중치 조합과 시드를 동시에 탐색
"""
import torch
import numpy as np
import itertools
import json
import os
from src import main, models, data, training

# 1. 탐색할 가중치 후보 (Label 1, Label 3 집중 탐색)
# Label 0, 2는 1.0으로 고정
l1_candidates = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]
l3_candidates = [1.2, 1.4, 1.6, 1.8, 2.0, 2.5]

# 2. 검증할 시드 (7시간 수면 시간 활용을 위해 시드 대폭 추가)
# 예상 소요 시간: 6 candidates * 6 candidates * 10 seeds * 1.2분 ≈ 7.2시간
seeds = [7, 42, 77, 123, 2026, 456, 789, 1004, 1, 3]

# 결과 저장
results = []
best_overall_loss = float('inf')
best_overall_acc = -1.0
best_config = None
best_model_params = None

print(f"{'='*70}")
print(f"  ⚖️  Optimal Weight Search Started")
print(f"  Combinations: {len(l1_candidates)} x {len(l3_candidates)} weights x {len(seeds)} seeds")
print(f"  Total Runs: {len(l1_candidates) * len(l3_candidates) * len(seeds)}")
print(f"{'='*70}\n")

# 데이터 로딩 (한 번만 수행)
config_base = main.Config()
data_loader = data.CompetitionDataLoader(
    data_dir=config_base.DATA_DIR,
    batch_size=config_base.BATCH_SIZE,
    shuffle=True
)
data_loader.load_data()
train_loader = data_loader.get_dataloader()

run_count = 0
total_runs = len(l1_candidates) * len(l3_candidates) * len(seeds)

for w1, w3 in itertools.product(l1_candidates, l3_candidates):
    current_weights = [1.0, w1, 1.0, w3]
    
    for seed in seeds:
        run_count += 1
        print(f"[{run_count}/{total_runs}] Weights={current_weights}, Seed={seed} ... ", end="", flush=True)
        
        # 설정 적용
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 모델 생성
        model = models.VQCClassifier(
            n_qubits=config_base.N_QUBITS,
            n_layers=config_base.N_LAYERS,
            topology=config_base.TOPOLOGY,
            measurement_qubits=config_base.MEASUREMENT_QUBITS,
            ansatz_type=config_base.ANSATZ_TYPE,
            measurement_type=config_base.MEASUREMENT_TYPE,
            init_scale=0.01
        )
        
        # 가중치 텐서 준비
        class_weights_tensor = torch.tensor(current_weights, dtype=torch.float32)
        
        # 훈련
        loss_history, acc_history, _ = training.train_model(
            model=model,
            train_loader=train_loader,
            epochs=config_base.EPOCHS,
            lr=config_base.LEARNING_RATE,
            loss_name='ce',
            optimizer_name='adam',
            device='cpu',
            verbose=False,
            weight_decay=0.0001,  # Regularization 유지
            smoothing=0.1,        # Smoothing 유지
            class_weights=class_weights_tensor
        )
        
        final_loss = loss_history[-1]
        final_acc = acc_history[-1]
        
        print(f"Loss={final_loss:.4f}, Acc={final_acc*100:.1f}%")
        
        # 결과 기록
        results.append({
            'weights': current_weights,
            'seed': seed,
            'loss': final_loss,
            'acc': final_acc
        })
        
        # 최고 기록 갱신 (정확도 100%이면서 Loss가 가장 낮은 것)
        # 만약 100%가 없다면, 정확도가 가장 높은 것 중 Loss가 낮은 것
        # [수정] 가중치가 다르면 Loss 스케일이 달라지므로, Accuracy를 최우선으로 비교해야 함
        if final_acc > best_overall_acc:
            best_overall_acc = final_acc
            best_overall_loss = final_loss
            best_config = {'weights': current_weights, 'seed': seed, 'acc': final_acc}
            best_model_params = model.params.detach().cpu().clone()
            print(f"   🏆 New Best Found! (Acc: {final_acc*100:.1f}%, Loss: {final_loss:.4f})")
        elif final_acc == best_overall_acc:
            if final_loss < best_overall_loss:
                best_overall_loss = final_loss
                best_config = {'weights': current_weights, 'seed': seed, 'acc': final_acc}
                best_model_params = model.params.detach().cpu().clone()
                print(f"   🏆 New Best Found! (Acc: {final_acc*100:.1f}%, Loss: {final_loss:.4f})")

print(f"\n{'='*70}")
print(f"  Search Complete")
print(f"{'='*70}")

if best_config:
    print(f"Best Configuration:")
    print(f"   Weights: {best_config['weights']}")
    print(f"   Seed: {best_config['seed']}")
    print(f"   Loss: {best_overall_loss:.4f}")
    print(f"   Acc: {best_config['acc']*100:.2f}%")

    # 결과 로그 파일 저장 (안전장치)
    with open('outputs/weight_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 전체 탐색 결과 저장: outputs/weight_search_results.json")

    # 최적 모델로 제출 파일 생성
    best_model = models.VQCClassifier(
        n_qubits=config_base.N_QUBITS,
        n_layers=config_base.N_LAYERS,
        topology=config_base.TOPOLOGY,
        measurement_qubits=config_base.MEASUREMENT_QUBITS,
        ansatz_type=config_base.ANSATZ_TYPE,
        measurement_type=config_base.MEASUREMENT_TYPE,  # [중요] 누락된 설정 추가
        init_scale=0.01
    )
    best_model.params.data = best_model_params
    main.create_submission(best_model, './outputs')
else:
    print("❌ 탐색 실패: 더 나은 모델을 찾지 못했습니다.")
