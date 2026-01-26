"""
측정 qubit 위치별 loss 탐색 (lr=0.035, epochs=220 고정)
[3,4], [6,7] 제외 26개 위치 자동 테스트
결과: outputs/measurement_search_results.json 저장
"""
import itertools
import json
from src import main, models, data, training, utils
import torch
import numpy as np

# 모든 2-qubit 측정 위치 (0~7 중 2개)
all_positions = list(itertools.combinations(range(8), 2))
tested = [[3,4], [6,7]]
remaining = [list(p) for p in all_positions if list(p) not in tested]

results = []

for pos in remaining:
    print(f"\n=== 측정 위치: {pos} ===")
    # Config 복사 및 측정 위치 변경
    config = main.Config()
    config.MEASUREMENT_QUBITS = pos
    config.LEARNING_RATE = 0.035
    config.EPOCHS = 220
    config.BATCH_SIZE = 4
    config.RANDOM_SEED = 42
    config.ANSATZ_TYPE = 'variational'
    config.TOPOLOGY = 'linear'
    config.OPTIMIZER = 'adam'
    config.LOSS_FUNCTION = 'ce'
    
    # 데이터 로딩 및 DataLoader 생성
    data_loader = data.CompetitionDataLoader(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    train_X, train_Y = data_loader.load_data()
    train_loader = data_loader.get_dataloader()
    t_train_X, t_train_Y = data_loader.get_torch_tensors()
    
    # 모델 생성
    model = models.VQCClassifier(
        n_qubits=config.N_QUBITS,
        n_layers=config.N_LAYERS,
        topology=config.TOPOLOGY,
        measurement_qubits=config.MEASUREMENT_QUBITS,
        ansatz_type=config.ANSATZ_TYPE,
        measurement_type=config.MEASUREMENT_TYPE,
        init_scale=0.01
    )
    
    # 훈련
    loss_history, acc_history, best_params = training.train_model(
        model=model,
        train_loader=train_loader,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        loss_name=config.LOSS_FUNCTION,
        optimizer_name=config.OPTIMIZER,
        device='cpu',
        verbose=False
    )
    final_loss = loss_history[-1]
    final_acc = acc_history[-1]
    best_loss = min(loss_history)
    
    print(f"측정 위치 {pos}: final_loss={final_loss:.4f}, best_loss={best_loss:.4f}, acc={final_acc*100:.2f}%")
    results.append({
        'measurement': pos,
        'final_loss': float(final_loss),
        'best_loss': float(best_loss),
        'final_acc': float(final_acc)
    })

# 결과 저장
with open('outputs/measurement_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== 모든 측정 위치 탐색 완료 ===")
print("outputs/measurement_search_results.json 저장됨")
