"""Analyze which 2 measured qubits carry the most class-discriminative signal.

Competition setting recap:
- Input is an 8-qubit pure state |psi> (statevector length 256).
- Only 2 qubits are measured -> 4 outcome probabilities -> 4-class classifier.

This script scores every qubit pair (i, j) by how separable the class-conditional
reduced density matrices are on those 2 qubits.

Score used (simple + robust for n=16):
- For each class c, compute mean reduced density matrix rho_c^{(i,j)} (4x4).
- Score(i,j) = average_{c<c'} TraceDistance(rho_c, rho_c')
  where TraceDistance(A,B) = 0.5 * ||A-B||_1.

Outputs:
- Prints top pairs.
- Writes detailed JSON to outputs/measurement_pair_analysis.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi)
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("Statevector has zero norm")
    return psi / norm


def reduced_density_matrix_pure_state(
    psi: np.ndarray, *, n_qubits: int, keep: Tuple[int, ...]
) -> np.ndarray:
    """Compute reduced density matrix on `keep` qubits for a pure state |psi>.

    psi: shape (2**n_qubits,), complex
    keep: tuple of qubit indices to keep (length k)

    Returns: rho_keep of shape (2**k, 2**k)
    """
    psi = _normalize_state(psi)

    keep = tuple(keep)
    if len(keep) == 0:
        raise ValueError("keep must be non-empty")
    if any(q < 0 or q >= n_qubits for q in keep):
        raise ValueError(f"keep contains invalid qubit index: {keep}")

    all_axes = list(range(n_qubits))
    trace_axes = [q for q in all_axes if q not in keep]

    # Reshape statevector into tensor with one axis per qubit.
    tensor = psi.reshape([2] * n_qubits)

    # Permute axes so kept qubits come first.
    perm = list(keep) + trace_axes
    tensor = np.transpose(tensor, axes=perm)

    dim_keep = 2 ** len(keep)
    dim_trace = 2 ** (n_qubits - len(keep))
    mat = tensor.reshape(dim_keep, dim_trace)

    # rho = mat @ mat^9
    rho = mat @ mat.conj().T
    return rho


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance for Hermitian matrices: 0.5 * sum(|eigvals(rho-sigma)|)."""
    delta = rho - sigma
    # Numerical symmetrization for stability
    delta = 0.5 * (delta + delta.conj().T)
    eigs = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(eigs)))


@dataclass
class PairResult:
    pair: Tuple[int, int]
    score_mean_pairwise_trace_distance: float
    per_class_probs: Dict[str, List[float]]


def marginal_probs_from_state(psi: np.ndarray, *, n_qubits: int, keep: Tuple[int, int]) -> np.ndarray:
    """Return measurement probabilities on two qubits (keep) in computational basis.

    This is the distribution WITHOUT any learned unitary; it's a useful baseline.
    Returns shape (4,) ordered as |00>,|01>,|10>,|11> on (keep[0], keep[1]).
    """
    psi = _normalize_state(psi)
    tensor = psi.reshape([2] * n_qubits)

    # Move kept axes to the end for easy reshape
    other = [q for q in range(n_qubits) if q not in keep]
    perm = other + list(keep)
    tensor = np.transpose(tensor, axes=perm)

    dim_other = 2 ** (n_qubits - 2)
    tensor = tensor.reshape(dim_other, 4)
    probs = np.sum(np.abs(tensor) ** 2, axis=0)
    return probs


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    x_path = os.path.join(root, "train_X.npy")
    y_path = os.path.join(root, "train_y.npy")

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 2:
        raise ValueError(f"train_X should be 2D (n, 256); got shape {X.shape}")

    n_samples, dim = X.shape
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(f"Statevector length {dim} is not a power of 2")

    classes = sorted(set(int(v) for v in y.tolist()))
    if len(classes) != 4:
        print(f"[WARN] Expected 4 classes; saw {classes}")

    # Pre-group samples by class
    by_class: Dict[int, List[np.ndarray]] = {c: [] for c in classes}
    for psi, label in zip(X, y):
        by_class[int(label)].append(psi)

    results: List[PairResult] = []

    for i, j in combinations(range(n_qubits), 2):
        pair = (i, j)

        # Compute class-mean reduced density matrices
        rho_mean: Dict[int, np.ndarray] = {}
        probs_mean: Dict[int, np.ndarray] = {}

        for c in classes:
            rhos = [reduced_density_matrix_pure_state(psi, n_qubits=n_qubits, keep=pair) for psi in by_class[c]]
            rho_mean[c] = np.mean(rhos, axis=0)

            probs = [marginal_probs_from_state(psi, n_qubits=n_qubits, keep=pair) for psi in by_class[c]]
            probs_mean[c] = np.mean(probs, axis=0)

        # Pairwise trace distances between class means
        dists = []
        for a_idx in range(len(classes)):
            for b_idx in range(a_idx + 1, len(classes)):
                ca = classes[a_idx]
                cb = classes[b_idx]
                dists.append(trace_distance(rho_mean[ca], rho_mean[cb]))

        score = float(np.mean(dists))

        per_class_probs = {str(c): probs_mean[c].tolist() for c in classes}
        results.append(
            PairResult(
                pair=pair,
                score_mean_pairwise_trace_distance=score,
                per_class_probs=per_class_probs,
            )
        )

    results.sort(key=lambda r: r.score_mean_pairwise_trace_distance, reverse=True)

    top_k = 10
    print(f"Analyzed {len(results)} pairs over n_qubits={n_qubits}.")
    print(f"Top {top_k} pairs by mean pairwise trace distance (higher is better):")
    for r in results[:top_k]:
        print(f"  pair={list(r.pair)} score={r.score_mean_pairwise_trace_distance:.6f}")

    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "measurement_pair_analysis.json")

    payload = {
        "n_qubits": n_qubits,
        "n_samples": int(n_samples),
        "classes": classes,
        "metric": "mean_pairwise_trace_distance_of_class_mean_reduced_density_matrices",
        "top": [
            {
                "pair": list(r.pair),
                "score": r.score_mean_pairwise_trace_distance,
                "per_class_mean_probs_without_unitary": r.per_class_probs,
            }
            for r in results
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
