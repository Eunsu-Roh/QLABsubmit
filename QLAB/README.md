# Quantum Phase Classification using Variational Quantum Circuits

**Score: 91.8** | 2nd Quantum AI Competition 2026

## Overview

This project implements a Variational Quantum Classifier (VQC) for quantum phase classification using PennyLane and PyTorch. The model achieves **91.8% accuracy** on the test dataset by optimizing circuit architecture and hyperparameters.

## Competition Task

- **Objective**: Classify 4 quantum phases (Cluster/SPT, Trivial, Ferromagnetic, Anti-ferromagnetic)
- **Training Data**: 16 labeled quantum states (256-dimensional complex vectors)
- **Constraint**: No data augmentation allowed (competition rule)

## Optimal Configuration

### Circuit Architecture
- **Qubits**: 8
- **Layers**: 5
- **Ansatz**: Variational (Hardware-Efficient Ansatz)
- **Topology**: Linear (7 CNOT gates per layer)
- **Measurement Qubits**: [3, 4]
- **Total Parameters**: 120

### Training Setup
- **Optimizer**: Adam
- **Learning Rate**: 0.035
- **Epochs**: 220
- **Batch Size**: 4
- **Loss Function**: Cross-Entropy
- **Random Seed**: 42 (reproducibility)

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 91.8% |
| Training Accuracy | 100% |
| Final Loss | 0.4133 |
| Best Loss | 0.4132 |

### Class-wise Performance
- **Cluster (SPT)**: 100%
- **Trivial**: 100%
- **Ferromagnetic**: 100%
- **Anti-ferromagnetic**: 100%

## Project Structure

```
.
├── src/
│   ├── main.py                 # Main execution script
│   ├── models/
│   │   └── vqc_classifier.py   # VQC model implementation
│   ├── quantum/
│   │   ├── circuit.py          # Quantum circuit builder
│   │   ├── ansatz.py           # Variational ansatz
│   │   └── feature_map.py      # Amplitude encoding
│   ├── training/
│   │   ├── optimizer.py        # Training loop & optimizers
│   │   └── loss.py             # Loss functions
│   ├── data/
│   │   ├── loader.py           # Data loader
│   │   └── preprocessing.py    # Data validation
│   └── utils/
│       ├── metrics.py          # Evaluation metrics
│       └── visualization.py    # Plotting functions
├── outputs/
│   └── submission.json         # Final submission (QASM + measurements)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create conda environment
conda create -n quantum-ml python=3.11
conda activate quantum-ml

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.11+
- PennyLane 0.39.0
- PyTorch 2.5.1
- NumPy, Matplotlib, Seaborn

## Usage

### Training & Evaluation

```bash
python src/main.py
```

This will:
1. Load training data (16 samples)
2. Initialize VQC with optimal configuration
3. Train for 220 epochs
4. Generate submission file (`outputs/submission.json`)
5. Save training history and confusion matrix

### Output Files
- `outputs/submission.json`: OpenQASM 2.0 circuit + measurement qubits
- `outputs/training_history.png`: Loss/accuracy curves
- `outputs/confusion_matrix.png`: Classification results
- `outputs/experiments.log`: Experiment records

## Key Design Decisions

### 1. Measurement Qubits Selection: [7, 8] → [3, 4]

**Initial Configuration**: Qubits [7, 8] (edge qubits)
**Final Configuration**: Qubits [3, 4] (central qubits)

**Theoretical Reasoning**:
- **Information Flow in Linear Topology**: In a linear CNOT chain (0→1→2→...→7), information propagates sequentially. Central qubits receive processed information from both directions, capturing more entanglement features.
- **Entanglement Distribution**: Edge qubits [7, 8] only interact with one neighboring qubit initially, while central qubits [3, 4] sit in the middle of the entanglement structure, accumulating correlations from multiple layers.
- **Phase Signature Locality**: Quantum phases manifest through multi-body correlations. Central measurements better capture global phase properties distributed across the circuit.
- **Expressivity vs. Generalization**: Edge qubits may learn features too specific to training data, while central qubits extract more robust, generalizable phase characteristics.

**Experimental Evidence**:
- Tested all 28 possible qubit pairs (8C2)
- Central pairs consistently outperformed edge pairs
- [3, 4] achieved lowest loss (0.4133) and highest test accuracy (91.8%)

### 2. Ansatz Choice
- **Variational (HEA)**: Best performance among tested options
- Compared: QCNN, StronglyEntangling, Hybrid
- Simple structure prevents overfitting with limited data (n=16)

### 3. Topology Optimization
- **Linear**: Outperforms ring, bidirectional, and all-to-all
- Fewer CNOT gates reduce circuit complexity
- Better suited for small training set

### 4. Hyperparameter Optimization (Manual Grid Search)

**Methodology**: Sequential manual optimization to isolate effects of each hyperparameter

**Step 1: Measurement Qubits** (Most Critical)
- Exhaustively tested all 28 pairs
- Selected [3, 4] based on lowest validation loss

**Step 2: Learning Rate** (Fine-grained tuning)
- Started with 0.05 (standard baseline)
- Reduced to 0.04 → Loss improved from 0.4196 to 0.4165
- Further reduced to 0.035 → Loss improved to 0.4133 (best)
- Smaller steps allowed more stable convergence

**Step 3: Epochs** (Convergence analysis)
- Extended from 200 to 220 epochs
- Monitored loss plateau to avoid overfitting
- 220 epochs achieved optimal train/test balance

**Step 4: Architecture** (Topology & Ansatz)
- Compared 4 topologies: Linear, Ring, Bidirectional, All-to-all
- Linear topology best suited for small training set (less overfitting)
- Variational ansatz outperformed QCNN, StronglyEntangling, and Hybrid

**Step 5: Batch Size** (Data efficiency)
- Tested batch sizes: 2, 4, 8, 16
- Batch size 4 optimal for 16 total samples (4 batches/epoch)

**Why Manual Optimization?**
- Small dataset (n=16) makes automated search unstable
- Each experiment is inexpensive (~2 minutes)
- Human intuition helps identify promising directions
- Avoids overfitting to validation noise in auto-tuning

## Experimental Results

| Experiment | Loss | Accuracy | Notes |
|-----------|------|----------|-------|
| Initial (lr=0.05) | 0.4196 | 100% | Baseline |
| lr=0.04 | 0.4165 | 100% | Improved |
| **lr=0.035** | **0.4133** | **100%** | **Best** |
| Topology: ring | 0.5189 | 93.75% | Worse |
| Topology: all | 0.5084 | 87.5% | Overfitting |
| Ansatz: QCNN | 0.4408 | 100% | Good but worse |
| Ansatz: SEL | 0.5160 | 93.75% | Overfitting |

## Competition Compliance

- **No Data Augmentation**: All augmentation code removed  
- **Reproducible**: Fixed random seed (42)  
- **Verifiable**: Complete source code from initialization to submission  
- **No Pseudo-labeling**: Only training data used

## Citation

```bibtex
@misc{quantum-phase-vqc-2026,
  title={Variational Quantum Circuit for Quantum Phase Classification},
  author={KAU Quantum AI Lab},
  year={2026},
  note={2nd Quantum AI Competition - Score: 91.8}
}
```

## Team

QLAB - KAU Quantum AI Lab  
Korea Aerospace University
