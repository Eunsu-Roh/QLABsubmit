"""\
QCNN LOOCV 스윕: (N_LAYERS in {3,4}) x (LR in {0.02, 0.05})

무엇을 하나요?
- train=16이라 train accuracy는 과적합 지표가 되기 쉽습니다.
- 그래서 LOOCV(Leave-One-Out CV) 평균/분산으로 구조를 선택하고,
    선택된 상위 1~2개 설정을 전체 데이터로 다시 학습한 뒤 제출 JSON을 저장합니다.

고정값
- measurement qubits: [5, 7]
- ansatz_type: 'qcnn'

실행:
    python qcnn_sweep.py

출력:
- outputs/qcnn_sweep_results.json  (LOOCV 결과 요약)
- outputs/submission_best1_*.json, outputs/submission_best2_*.json
- outputs/submission.json (best1으로 자동 업데이트)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO

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
class SweepConfig:
    n_layers: int
    lr: float


def create_submission(model: VQCClassifier) -> dict:
    import pennylane as qml

    qc = model.get_quantum_circuit()
    trained_params = model.params.detach().cpu()
    ansatz_circuit = qc.get_ansatz_circuit()

    qasm_string = qml.to_openqasm(ansatz_circuit, measure_all=False)(trained_params)
    return {"measurements": qc.measurement_qubits, "qasm": qasm_string}


def _make_loader(tX: torch.Tensor, tY: torch.Tensor, indices: np.ndarray, batch_size: int = 4) -> DataLoader:
    ds = TensorDataset(tX[indices], tY[indices])
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def _eval_one(model: VQCClassifier, x: torch.Tensor, y: torch.Tensor, device: str) -> int:
    model.eval()
    with torch.no_grad():
        probs = model(x.to(device))
        pred = torch.argmax(probs, dim=1).cpu()
    return int((pred == y).item())


def run_loocv(cfg: SweepConfig, tX: torch.Tensor, tY: torch.Tensor, device: str, seeds: list[int], epochs: int) -> dict:
    n = tX.shape[0]
    acc_per_seed: list[float] = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        correct = 0
        for held_out in range(n):
            train_idx = np.array([i for i in range(n) if i != held_out], dtype=np.int64)
            loader = _make_loader(tX, tY, train_idx, batch_size=4)

            # Suppress very verbose model/circuit init prints during LOOCV.
            with redirect_stdout(StringIO()):
                model = VQCClassifier(
                    n_qubits=8,
                    n_layers=cfg.n_layers,
                    topology='linear',
                    measurement_qubits=[5, 7],
                    ansatz_type='qcnn',
                    init_scale=0.01,
                    measurement_type='computational',
                )

                train_model(
                    model=model,
                    train_loader=loader,
                    epochs=epochs,
                    lr=cfg.lr,
                    loss_name='ce',
                    optimizer_name='adam',
                    device=device,
                    verbose=False,
                )

            x_ho = tX[held_out : held_out + 1]
            y_ho = tY[held_out : held_out + 1]
            correct += _eval_one(model, x_ho, y_ho, device)

        acc_per_seed.append(correct / n)

    mean_acc = float(np.mean(acc_per_seed))
    std_acc = float(np.std(acc_per_seed))

    return {
        'name': f"qcnn_L{cfg.n_layers}_lr{cfg.lr:g}",
        'config': {'ansatz_type': 'qcnn', 'n_layers': cfg.n_layers, 'lr': cfg.lr, 'measurement_qubits': [5, 7]},
        'seeds': seeds,
        'loocv_epochs': epochs,
        'loocv_acc_per_seed': acc_per_seed,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
    }


def train_full_and_export(cfg: SweepConfig, seed: int, out_dir: str) -> str:
    """Train on full 16 samples and export a submission JSON. Returns saved path."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dl = CompetitionDataLoader(data_dir='./', batch_size=4, shuffle=True)
    dl.load_data()
    train_loader = dl.get_dataloader()

    model = VQCClassifier(
        n_qubits=8,
        n_layers=cfg.n_layers,
        topology='linear',
        measurement_qubits=[5, 7],
        ansatz_type='qcnn',
        init_scale=0.01,
        measurement_type='computational',
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(
        model=model,
        train_loader=train_loader,
        epochs=200,
        lr=cfg.lr,
        loss_name='ce',
        optimizer_name='adam',
        device=device,
        verbose=False,
    )

    submission = create_submission(model)
    name = f"qcnn_L{cfg.n_layers}_lr{cfg.lr:g}_seed{seed}"
    path = os.path.join(out_dir, f"submission_{name}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2)
    return path

def main():
    parser = argparse.ArgumentParser(description='QCNN LOOCV sweep for few-shot model selection')
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run a much faster LOOCV (fewer epochs/seeds/configs) to get a good-enough best1/best2 quickly.',
    )
    parser.add_argument('--loocv-epochs', type=int, default=None, help='Override LOOCV epochs.')
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='Comma-separated seeds for LOOCV, e.g. "42,123,789". Default depends on --quick.',
    )
    parser.add_argument('--full-epochs', type=int, default=200, help='Epochs for final full-data training.')
    args = parser.parse_args()

    out_dir = './outputs'
    os.makedirs(out_dir, exist_ok=True)

    if args.quick:
        # Fast preset: 2 configs x 1 seed x 16 folds x 40 epochs = 32 short trainings.
        sweep = [
            SweepConfig(n_layers=3, lr=0.02),
            SweepConfig(n_layers=4, lr=0.02),
        ]
        default_seeds = [42]
        default_loocv_epochs = 40
    else:
        sweep = [
            SweepConfig(n_layers=3, lr=0.02),
            SweepConfig(n_layers=3, lr=0.05),
            SweepConfig(n_layers=4, lr=0.02),
            SweepConfig(n_layers=4, lr=0.05),
        ]
        default_seeds = [42, 123, 789]
        default_loocv_epochs = 120  # 속도/안정성 균형. 필요시 200으로 올려도 됨.

    # Load tensors once
    dl = CompetitionDataLoader(data_dir='./', batch_size=4, shuffle=True)
    dl.load_data()
    tX, tY = dl.get_torch_tensors()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # LOOCV settings
    if args.seeds is None:
        seeds = default_seeds
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        if not seeds:
            raise ValueError('--seeds provided but empty')

    loocv_epochs = args.loocv_epochs if args.loocv_epochs is not None else default_loocv_epochs

    print('[INFO] sweep configs:', [f"L{c.n_layers}_lr{c.lr:g}" for c in sweep])
    print(f"[INFO] measurement qubits: [5, 7]")
    print(f"[INFO] LOOCV seeds={seeds} epochs={loocv_epochs} (quick={args.quick})")

    loocv_results = []
    for cfg in sweep:
        name = f"qcnn_L{cfg.n_layers}_lr{cfg.lr:g}"
        print('\n' + '=' * 70)
        print(f"LOOCV {name}")
        print('=' * 70)

        res = run_loocv(cfg, tX, tY, device, seeds=seeds, epochs=loocv_epochs)
        loocv_results.append(res)
        print(f"mean={res['mean_acc']*100:.2f}% std={res['std_acc']*100:.2f}%  per_seed={res['loocv_acc_per_seed']}")

    # rank: mean desc, std asc
    ranked = sorted(loocv_results, key=lambda r: (-r['mean_acc'], r['std_acc']))

    summary_path = os.path.join(out_dir, 'qcnn_sweep_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'results': ranked}, f, indent=2)

    # pick top-2
    top_cfgs = ranked[:2]
    print('\n' + '=' * 70)
    print('TOP-2 by LOOCV')
    print('=' * 70)
    for i, r in enumerate(top_cfgs, 1):
        print(f"#{i} {r['name']}  mean={r['mean_acc']*100:.2f}% std={r['std_acc']*100:.2f}%")

    # Train full and export submissions for top-2
    saved_paths: list[str] = []
    for i, r in enumerate(top_cfgs, 1):
        cfg = SweepConfig(n_layers=int(r['config']['n_layers']), lr=float(r['config']['lr']))
        # Use a fixed seed for reproducibility of the exported submission.
        torch.manual_seed(42)
        np.random.seed(42)

        # Train full-data model and export.
        # (We keep final training longer than LOOCV to fit the tiny train set.)
        path = train_full_and_export(cfg, seed=42, out_dir=out_dir)
        best_path = os.path.join(out_dir, f"submission_best{i}_{os.path.basename(path).replace('submission_', '')}")
        os.replace(path, best_path)
        saved_paths.append(best_path)
        print(f"saved best{i}: {best_path}")

    # Update default submission.json to best1
    if saved_paths:
        with open(saved_paths[0], 'r', encoding='utf-8') as f:
            best_submission = json.load(f)
        with open(os.path.join(out_dir, 'submission.json'), 'w', encoding='utf-8') as f:
            json.dump(best_submission, f, indent=2)
        print(f"updated: {os.path.join(out_dir, 'submission.json')}")

    print('\n' + '=' * 70)
    print('DONE')
    print('=' * 70)
    print(f"saved: {summary_path}")
    print("best submissions: outputs/submission_best1_*.json / outputs/submission_best2_*.json")


if __name__ == '__main__':
    main()
