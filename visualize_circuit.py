#!/usr/bin/env python3
"""
Visualize the QUAFFLE per-group quantum circuit (PennyLane backend).

The live circuit is defined by `pennylane_default_circuit` inside
utilsTorch/unetTorch.py. To avoid importing the full model stack (which pulls
in PyTorch CUDA, PennyLane-Lightning, etc.), this script rebuilds an
EQUIVALENT circuit using the same gate sequence and settings.

--q-ansatz reupload (default, pennylane_reupload_circuit), per layer l:
      1. Encoding        RY(pi * tanh(s_lj * x_j)) on wire j (s_lj learnable, shown as 1)
      2. Variational     Rot(w_lj0, w_lj1, w_lj2) on wire j
      3. Entangling      CNOT ring 0->1->2->3->0
    Measurement: <Z_j> on each wire.

--q-ansatz legacy (original model, pennylane_default_circuit).
    Per 4-channel group (4 qubits):
      1. Data encoding   — RX(x_i) on wire i, for i in 0..3
      2. Variational     — RY(theta_i) on wire i, for i in 0..3
      3. Entangling      — linear CNOT chain 0->1->2->3
                          + ring closure CNOT(3->0)
      4. Measurement     — <Z_i> expval on each wire

Reference in codebase: pennylane_default_circuit() and
pennylane_reupload_circuit() in utilsTorch/unetTorch.py. See docs/MODEL.md.

Training uses q_backend='torch' by default, an exact torch statevector
simulation of the same circuits (torch_legacy_expvals and
torch_reupload_expvals in utilsTorch/unetTorch.py). It gives the same <Z>
values as the PennyLane QNodes drawn here, see tests/test_quantum_backend.py.

Outputs:
    results/plots/quantum_circuit.txt        — text diagram (qml.draw)
    results/plots/quantum_circuit.png        — matplotlib diagram (qml.draw_mpl)
    results/plots/quantum_circuit_bloch.png  — per-qubit <Z> bar chart with
                                                sample input/params (optional)

Usage:
    python visualize_circuit.py [--q-ansatz reupload|legacy] [--q-layers 2]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pennylane as qml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "plots")

N_QUBITS = 4  # QUAFFLE fixes groups to 4 channels / 4 qubits


def build_reupload_circuit(n_qubits: int = N_QUBITS, n_layers: int = 2):
    """Return a QNode that mirrors pennylane_reupload_circuit for one sample.

    Signature: circuit(x, params) with params of shape (n_layers, n_qubits, 3).
    Input scales are fixed to their initial value 1.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, params):
        for l in range(n_layers):
            for j in range(n_qubits):
                qml.RY(np.pi * np.tanh(x[j]), wires=j)
            for j in range(n_qubits):
                qml.Rot(params[l, j, 0], params[l, j, 1], params[l, j, 2], wires=j)
            for j in range(n_qubits):
                qml.CNOT(wires=[j, (j + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    return circuit


def build_circuit(n_qubits: int = N_QUBITS):
    """Return a PennyLane QNode that mirrors pennylane_default_circuit.

    Signature: circuit(x, params) where
        x      : array-like of length n_qubits (per-channel input values)
        params : array-like of length n_qubits (variational angles)
    Returns a list of n_qubits <Z_i> expectation values.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, params):
        # 1. Data encoding: one RX per qubit
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
        # 2. Variational: one RY per qubit
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        # 3. Entangling: linear chain + ring closure
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
        # 4. Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def _sample_inputs(seed: int = 42, ansatz: str = "legacy", n_layers: int = 2):
    """Reasonable example values to show a non-trivial circuit evaluation.

    Inputs (x) match the range of spatial averages after upstream
    normalization/activations — arbitrary small angles are fine for display.
    Params follow the initialization used in QuantumBlock: small Gaussian.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(low=-1.0, high=1.0, size=N_QUBITS)
    if ansatz == "reupload":
        # Reupload init: uniform in [-pi, pi]
        params = rng.uniform(low=-np.pi, high=np.pi, size=(n_layers, N_QUBITS, 3))
        return x.astype(float), params.astype(float)
    params = rng.normal(loc=0.0, scale=0.1, size=N_QUBITS)
    return x.astype(float), params.astype(float)


def save_text_diagram(circuit, x, params, path):
    text = qml.draw(circuit, level="device", decimals=3)(x, params)
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


def save_mpl_diagram(circuit, x, params, path):
    fig, ax = qml.draw_mpl(circuit, decimals=3, style="pennylane")(x, params)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_expval_barplot(circuit, x, params, path):
    """Bar chart of the per-qubit <Z> output for the sample input."""
    vals = np.array(circuit(x, params), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [f"q{i}\n<Z>" for i in range(N_QUBITS)],
        vals,
        color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"][:N_QUBITS],
        edgecolor="white",
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.02 if v >= 0 else -0.05),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Expectation value")
    ax.set_title(
        "Sample output of the per-group circuit\n"
        f"x={np.round(x, 3).tolist()}"
        + (f"   θ={np.round(params, 3).tolist()}" if np.ndim(params) == 1 else f"   θ: {np.shape(params)} angles"),
        fontsize=10,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return vals


def main():
    parser = argparse.ArgumentParser(description="Visualize the QUAFFLE quantum circuit.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for sample input/params (default: 42)")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--q-ansatz", choices=["reupload", "legacy"], default="reupload",
                        help="Circuit to draw (default: reupload)")
    parser.add_argument("--q-layers", type=int, default=2,
                        help="Re-uploading layers for --q-ansatz reupload (default: 2)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.q_ansatz == "reupload":
        circuit = build_reupload_circuit(N_QUBITS, args.q_layers)
        suffix = f"_reupload_L{args.q_layers}"
    else:
        circuit = build_circuit(N_QUBITS)
        suffix = ""
    x, params = _sample_inputs(seed=args.seed, ansatz=args.q_ansatz, n_layers=args.q_layers)

    txt_path = os.path.join(args.out_dir, f"quantum_circuit{suffix}.txt")
    png_path = os.path.join(args.out_dir, f"quantum_circuit{suffix}.png")
    bar_path = os.path.join(args.out_dir, f"quantum_circuit{suffix}_expvals.png")

    text = save_text_diagram(circuit, x, params, txt_path)
    save_mpl_diagram(circuit, x, params, png_path)
    vals = save_expval_barplot(circuit, x, params, bar_path)

    print(text)
    print()
    print(f"Saved: {txt_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {bar_path}")
    print(f"Sample <Z> outputs: {np.round(vals, 4).tolist()}")


if __name__ == "__main__":
    main()
