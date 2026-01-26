"""
Grid Search for lr and epochs fine-tuning
"""

import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, '.')

from src.models import VQCClassifier
from src.data import CompetitionDataLoader
from src.training import train_model

# Grid search parameters
lr_candidates = [0.033, 0.034, 0.036, 0.037]
epochs_candidates = [215, 217, 223, 225]

# Fixed configuration
N_QUBITS = 8
N_LAYERS = 5
TOPOLOGY = 'linear'
MEASUREMENT_QUBITS = [3, 4]
ANSATZ_TYPE = 'variational'
BATCH_SIZE = 4
RANDOM_SEED = 42

# Results storage
results = []

print("="*70)
print("  Grid Search: lr × epochs Fine-tuning")
print("="*70)
print(f"Total experiments: {len(lr_candidates)} × {len(epochs_candidates)} = {len(lr_candidates) * len(epochs_candidates)}")
print()

# Load data once
data_loader = CompetitionDataLoader(
    data_dir='./',
    batch_size=BATCH_SIZE,
    shuffle=True
)
train_X, train_Y = data_loader.load_data()
train_loader = data_loader.get_dataloader()

experiment_num = 0
total_experiments = len(lr_candidates) * len(epochs_candidates)

for lr in lr_candidates:
    for epochs in epochs_candidates:
        experiment_num += 1
        
        print(f"\n[{experiment_num}/{total_experiments}] lr={lr}, epochs={epochs}")
        print("-" * 50)
        
        # Set seed
        import torch
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        # Create model
        model = VQCClassifier(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            topology=TOPOLOGY,
            measurement_qubits=MEASUREMENT_QUBITS,
            ansatz_type=ANSATZ_TYPE
        )
        
        # Train
        loss_history, acc_history, best_params = train_model(
            model=model,
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            loss_name='ce',
            optimizer_name='adam',
            device='cpu',
            verbose=False
        )
        
        final_loss = loss_history[-1]
        final_acc = acc_history[-1]
        best_loss = min(loss_history)
        
        # Store result
        result = {
            'lr': lr,
            'epochs': epochs,
            'final_loss': float(final_loss),
            'final_acc': float(final_acc),
            'best_loss': float(best_loss),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        results.append(result)
        
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Final Acc:  {final_acc:.4f}")
        print(f"  Best Loss:  {best_loss:.4f}")

print("\n" + "="*70)
print("  Results Summary")
print("="*70)

# Sort by best loss
results_sorted = sorted(results, key=lambda x: x['best_loss'])

print(f"\n{'Rank':<6}{'LR':<8}{'Epochs':<8}{'Best Loss':<12}{'Final Loss':<12}{'Final Acc':<10}")
print("-" * 70)

for i, r in enumerate(results_sorted[:10]):
    rank = f"#{i+1}"
    print(f"{rank:<6}{r['lr']:<8}{r['epochs']:<8}{r['best_loss']:<12.4f}{r['final_loss']:<12.4f}{r['final_acc']:<10.2%}")

# Save results
with open('outputs/grid_search_results.json', 'w') as f:
    json.dump(results_sorted, f, indent=2)

print(f"\n💾 Results saved to: outputs/grid_search_results.json")

# Best configuration
best = results_sorted[0]
print(f"\n🏆 Best Configuration:")
print(f"   lr = {best['lr']}")
print(f"   epochs = {best['epochs']}")
print(f"   best_loss = {best['best_loss']:.4f}")

# Compare with current best (0.4133)
current_best = 0.4133
improvement = (current_best - best['best_loss']) / current_best * 100

if best['best_loss'] < current_best:
    print(f"\n✅ Improvement: {improvement:.2f}% better than current best!")
else:
    print(f"\n❌ No improvement: {abs(improvement):.2f}% worse")

print("="*70)
