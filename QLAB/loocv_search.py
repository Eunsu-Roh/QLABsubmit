"""\
LOOCV(Leave-One-Out CV) 기반 간단한 구조/측정 설정 탐색.

왜 필요한가?
- train=16이라 train accuracy는 거의 항상 100%가 됩니다.
- 구조 선택은 '일반화 안정성'을 봐야 하므로 LOOCV가 가장 안전한 기본 도구입니다.

사용 예:
  python loocv_search.py

결과는 outputs/loocv_results.json 에 저장됩니다.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 프로젝트 루트를 path에 추가 (root에서 실행 시 필요)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data import CompetitionDataLoader
from src.models import VQCClassifier
from src.training import train_model


@dataclass(frozen=True)
class Candidate:
    name: str
    ansatz_type: str
    n_layers: int
    topology: str
    measurement_qubits: list[int]
    lr: float = 0.05
    epochs: int = 200
    init_scale: float = 0.01


def _make_loader(tX: torch.Tensor, tY: torch.Tensor, indices: np.ndarray, batch_size: int = 4) -> DataLoader:
    ds = TensorDataset(tX[indices], tY[indices])
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def _eval_one(model: VQCClassifier, x: torch.Tensor, y: torch.Tensor, device: str) -> int:
    model.eval()
    with torch.no_grad():
        probs = model(x.to(device))
        pred = torch.argmax(probs, dim=1).cpu()
    return int((pred == y).item())


def run_loocv(cand: Candidate, tX: torch.Tensor, tY: torch.Tensor, device: str, seeds: list[int]) -> dict:
    n = tX.shape[0]
    fold_correct = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        correct = 0
        for held_out in range(n):
            train_idx = np.array([i for i in range(n) if i != held_out], dtype=np.int64)

            loader = _make_loader(tX, tY, train_idx, batch_size=4)

            model = VQCClassifier(
                n_qubits=8,
                n_layers=cand.n_layers,
                topology=cand.topology,
                measurement_qubits=cand.measurement_qubits,
                ansatz_type=cand.ansatz_type,
                init_scale=cand.init_scale,
                measurement_type='computational',
            )
            train_model(
                model=model,
                train_loader=loader,
                epochs=cand.epochs,
                lr=cand.lr,
                loss_name='ce',
                optimizer_name='adam',
                device=device,
                verbose=False,
            )

            x_ho = tX[held_out : held_out + 1]
            y_ho = tY[held_out : held_out + 1]
            correct += _eval_one(model, x_ho, y_ho, device)

        fold_correct.append(correct / n)

    return {
        'name': cand.name,
        'config': {
            'ansatz_type': cand.ansatz_type,
            'n_layers': cand.n_layers,
            'topology': cand.topology,
            'measurement_qubits': cand.measurement_qubits,
            'lr': cand.lr,
            'epochs': cand.epochs,
            'init_scale': cand.init_scale,
        },
        'seeds': seeds,
        'loocv_acc_per_seed': fold_correct,
        'mean_acc': float(np.mean(fold_correct)),
        'std_acc': float(np.std(fold_correct)),
    }


def main():
    out_dir = './outputs'
    os.makedirs(out_dir, exist_ok=True)

    # 데이터 로드
    data_loader = CompetitionDataLoader(data_dir='./', batch_size=4, shuffle=True)
    data_loader.load_data()
    tX, tY = data_loader.get_torch_tensors()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 탐색 후보
    candidates = [
        Candidate(
            name='qcnn_meas_3_7_L2',
            ansatz_type='qcnn',
            n_layers=2,
            topology='linear',
            measurement_qubits=[3, 7],
            lr=0.03,
            epochs=200,
            init_scale=0.01,
        ),
        Candidate(
            name='qcnn_meas_3_7_L3',
            ansatz_type='qcnn',
            n_layers=3,
            topology='linear',
            measurement_qubits=[3, 7],
            lr=0.03,
            epochs=200,
            init_scale=0.01,
        ),
        Candidate(
            name='two_local_center_L4',
            ansatz_type='two_local',
            n_layers=4,
            topology='linear',
            measurement_qubits=[4, 5],
            lr=0.05,
            epochs=200,
            init_scale=0.01,
        ),
        Candidate(
            name='alt_hybrid_baseline',
            ansatz_type='alt_hybrid',
            n_layers=5,
            topology='linear',
            measurement_qubits=[6, 7],
            lr=0.05,
            epochs=200,
            init_scale=0.01,
        ),
    ]

    seeds = [42, 123, 789]

    results = []
    for cand in candidates:
        print('\n' + '=' * 70)
        print(f"LOOCV: {cand.name}")
        print('=' * 70)
        res = run_loocv(cand, tX, tY, device, seeds)
        results.append(res)
        print(f"mean={res['mean_acc']*100:.2f}% std={res['std_acc']*100:.2f}%  per_seed={res['loocv_acc_per_seed']}")

    results_sorted = sorted(results, key=lambda r: r['mean_acc'], reverse=True)

    out_path = os.path.join(out_dir, 'loocv_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results_sorted}, f, indent=2)

    best = results_sorted[0]
    print('\n' + '=' * 70)
    print('BEST (by LOOCV mean)')
    print('=' * 70)
    print(best['name'])
    print(best['config'])
    print(f"mean={best['mean_acc']*100:.2f}% std={best['std_acc']*100:.2f}%")
    print(f"saved: {out_path}")


if __name__ == '__main__':
    main()
