"""
Random Seed Search (Best Model Selection)
더 넓은 범위의 Seed 탐색을 통해 최적의 초기값(Lucky Seed) 발견
outputs/ensemble_log.txt, outputs/submission.json 생성
"""
import torch
import numpy as np
from src import main, models, data, training, utils

# 더 넓은 범위의 시드 탐색 (Top 10 진입을 위한 Lucky Seed 찾기)
seeds = [
    42, 123, 456, 789, 7, 77, 777, 888, 999, 1000, 
    2025, 2026, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
]
loss_list = []
acc_list = []

# Best Model Tracking
best_overall_acc = -1.0
best_overall_loss = float('inf')
best_model_params = None
best_seed = None

# 데이터 로딩 (루프 밖에서 한 번만 실행하여 속도 향상)
config_init = main.Config()
data_loader = data.CompetitionDataLoader(
    data_dir=config_init.DATA_DIR,
    batch_size=config_init.BATCH_SIZE,
    shuffle=True
)
train_X, train_Y = data_loader.load_data()
train_loader = data_loader.get_dataloader()
t_train_X, t_train_Y = data_loader.get_torch_tensors()

for seed in seeds:
    print(f"\n=== Seed: {seed} ===")
    config = main.Config()
    
    config.RANDOM_SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    
    # 초기화 다양성 확인 (디버깅)
    init_param_sum = model.params.sum().item()
    print(f"   [Check] Init params sum: {init_param_sum:.6f}")
    
    device = 'cpu'
    
    # Class weights preparation
    class_weights = None
    if getattr(config, 'USE_CLASS_WEIGHTS', False):
        class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
        if seed == seeds[0]: print(f"   ⚖️  Class Weights Active: {config.CLASS_WEIGHTS}")
    
    # 훈련
    loss_history, acc_history, best_params = training.train_model(
        model=model,
        train_loader=train_loader,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        loss_name=config.LOSS_FUNCTION,
        optimizer_name=config.OPTIMIZER,
        device=device,
        verbose=False,
        scheduler_name=getattr(config, 'SCHEDULER', None),
        weight_decay=getattr(config, 'WEIGHT_DECAY', 0.0),
        smoothing=getattr(config, 'SMOOTHING', 0.0),
        class_weights=class_weights,
        gamma=getattr(config, 'FOCAL_GAMMA', 2.0)
    )
    
    final_loss = loss_history[-1]
    final_acc = acc_history[-1]
    
    loss_list.append(final_loss)
    acc_list.append(final_acc)
    print(f"Seed {seed}: final_loss={final_loss:.4f}, acc={final_acc*100:.2f}%")

    # Update Best Model
    if final_acc > best_overall_acc or (final_acc == best_overall_acc and final_loss < best_overall_loss):
        best_overall_acc = final_acc
        best_overall_loss = final_loss
        best_model_params = model.params.detach().cpu().clone()
        best_seed = seed
        print(f"   🏆 New Best Model found! (Seed {seed})")

# 최적 모델 로드
print(f"\n=== 최적 모델 선택 결과 ===")
print(f"Best Seed: {best_seed}")
print(f"Best Accuracy: {best_overall_acc*100:.2f}%")
print(f"Best Loss: {best_overall_loss:.4f}")

config = main.Config()
best_model = models.VQCClassifier(
    n_qubits=config.N_QUBITS,
    n_layers=config.N_LAYERS,
    topology=config.TOPOLOGY,
    measurement_qubits=config.MEASUREMENT_QUBITS,
    ansatz_type=config.ANSATZ_TYPE,
    measurement_type=config.MEASUREMENT_TYPE,
    init_scale=0.01
)
best_model.params.data = best_model_params

# 평가
pred = best_model(t_train_X)
pred_labels = pred.argmax(dim=1)
acc = (pred_labels == t_train_Y).float().mean().item()

print(f"\n=== 최종 선택 모델 검증 ===")
print(f"Verify Accuracy: {acc*100:.2f}%")
print(f"Best Seed used: {best_seed}")

# 제출 파일 생성
main.create_submission(best_model, './outputs')

# 로그 저장
with open('outputs/ensemble_log.txt', 'w') as f:
    f.write(f"Seeds: {seeds}\n")
    f.write(f"Losses: {loss_list}\n")
    f.write(f"Accuracies: {acc_list}\n")
    f.write(f"Best Seed: {best_seed}\n")
    f.write(f"Best Train Accuracy: {acc*100:.2f}%\n")

print("outputs/submission.json, outputs/ensemble_log.txt 생성 완료")
