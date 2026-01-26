"""\
Quantum Convolutional Neural Network (QCNN) style ansatz.

Design goals for this competition setting:
- Only 2 qubits are measured -> information must be compressed.
- Only 16 training samples -> aggressively reduce parameters and share weights.
- OpenQASM 2 export friendly -> use only RY/RZ/CNOT.

This ansatz implements a simple hierarchical compression for 8 qubits:
    (0,1)(2,3)(4,5)(6,7)  --pool-->  (1,3)(5,7)  --pool-->  (5,7)

Recommended measurement qubits: [5, 7]
"""

from __future__ import annotations

import pennylane as qml


class QCNNAnsatz:
    """\
    QCNN-style hierarchical ansatz.

    Parameters are shared within each "convolution layer" to reduce overfitting.

    Notes:
    - This implementation is intentionally conservative for generalization.
    - It is tailored to n_qubits=8 (competition constraint).
    """

    # per two-qubit block parameters
    _BLOCK_PARAMS = 6

    def __init__(self, n_qubits: int = 8, n_layers: int = 2):
        """
        Args:
            n_qubits: number of qubits (must be 8 for this competition).
            n_layers: number of bottom "convolution" layers (recommended 1~4).
        """
        if n_qubits != 8:
            raise ValueError("QCNNAnsatz currently supports n_qubits=8 only")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Total params:
        # - bottom conv layers: n_layers blocks (shared across qubit pairs)
        # - level-1 conv: 1 block param set (shared across 2 blocks)
        # - level-2 conv: 1 block param set (single block)
        # - readout rotations on (5,7): 4 params
        self.n_params = self._BLOCK_PARAMS * (self.n_layers + 2) + 4

    def _two_qubit_block(self, block_params, wire_a: int, wire_b: int):
        """A small expressive 2-qubit block using only RY/RZ/CNOT."""
        # block_params shape: (6,)
        qml.RY(block_params[0], wires=wire_a)
        qml.RY(block_params[1], wires=wire_b)
        qml.CNOT(wires=[wire_a, wire_b])
        qml.RZ(block_params[2], wires=wire_a)
        qml.RZ(block_params[3], wires=wire_b)
        qml.CNOT(wires=[wire_b, wire_a])
        qml.RY(block_params[4], wires=wire_a)
        qml.RY(block_params[5], wires=wire_b)

    def _pool_pair(self, control: int, target: int):
        """Pooling-like compression: push information from control into target."""
        qml.CNOT(wires=[control, target])

    def apply(self, params):
        """Apply QCNN ansatz.

        Args:
            params: flat parameter tensor of shape (n_params,)
        """
        # Split parameters
        idx = 0

        # Bottom conv layers (alternating disjoint pairing)
        bottom = params[idx : idx + self._BLOCK_PARAMS * self.n_layers]
        idx += self._BLOCK_PARAMS * self.n_layers
        bottom = bottom.reshape(self.n_layers, self._BLOCK_PARAMS)

        # Level-1 / Level-2 conv blocks
        level1 = params[idx : idx + self._BLOCK_PARAMS]
        idx += self._BLOCK_PARAMS
        level2 = params[idx : idx + self._BLOCK_PARAMS]
        idx += self._BLOCK_PARAMS

        # Readout rotations (on wires 5 and 7)
        readout = params[idx : idx + 4]

        for layer in range(self.n_layers):
            # Alternate pairing to improve mixing while staying local.
            if layer % 2 == 0:
                pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
            else:
                pairs = [(1, 2), (3, 4), (5, 6)]

            for a, b in pairs:
                self._two_qubit_block(bottom[layer], a, b)

        # Pool to odd wires: 0->1, 2->3, 4->5, 6->7
        self._pool_pair(0, 1)
        self._pool_pair(2, 3)
        self._pool_pair(4, 5)
        self._pool_pair(6, 7)

        # Level-1 conv on (1,3) and (5,7) with shared parameters
        self._two_qubit_block(level1, 1, 3)
        self._two_qubit_block(level1, 5, 7)

        # Pool to (5,7)
        # Push information from (1,3) into (5,7) so the final 2-qubit readout
        # can focus on wires (5,7).
        self._pool_pair(1, 5)
        self._pool_pair(3, 7)

        # Level-2 conv on (5,7)
        self._two_qubit_block(level2, 5, 7)

        # Final readout basis tuning (computational probs)
        qml.RY(readout[0], wires=5)
        qml.RZ(readout[1], wires=5)
        qml.RY(readout[2], wires=7)
        qml.RZ(readout[3], wires=7)

    def __repr__(self):
        return f"QCNNAnsatz(n_qubits={self.n_qubits}, n_layers={self.n_layers}, n_params={self.n_params})"
