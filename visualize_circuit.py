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

Training uses q_backend='pennylane' by default (batched PennyLane QNode with
backprop differentiation), with an optional exact torch statevector simulator
(q_backend='torch', torch_legacy_expvals and torch_reupload_expvals in
utilsTorch/unetTorch.py) that produces identical <Z> values; see
tests/test_quantum_backend.py.

Outputs:
    results/plots/quantum_circuit.txt                         — text diagram (qml.draw)
    results/plots/quantum_circuit.png                         — matplotlib diagram (qml.draw_mpl)
    results/plots/quantum_circuit_expvals.png                 — per-qubit <Z> bar chart
    results/plots/quaffle_circuits_legacy_vs_reupload.{png,pdf} — side-by-side comparison (--q-ansatz compare)

Usage:
    python visualize_circuit.py [--q-ansatz reupload|legacy|compare] [--q-layers 2]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Arc

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


def _hqunet_grid_edges(grid: int = 2):
    G = int(grid)
    if G not in (2, 4):
        raise ValueError(f"hqunet_grid must be 2 or 4, got {grid!r}")
    row_edges = [(r * G + c, r * G + c + 1) for r in range(G) for c in range(G - 1)]
    col_edges = [(r * G + c, (r + 1) * G + c) for c in range(G) for r in range(G - 1)]
    return row_edges, col_edges


def build_hqunet_circuit(grid: int = 2, n_layers: int = 2, return_state: bool = False):
    """Return a PennyLane QNode mirroring pennylane_hqunet_state_qnode (#18).

    Signature: circuit(x, params) where
        x      : array of shape (G*G, 3) with 3-axis (RX, RY, RZ) patch angles
        params : array of shape (n_layers, 2, 4) with shared horizontal (F_R)
                 and vertical (F_C) 2-qubit filter angles per layer.
    """
    G = int(grid)
    n_qubits = G * G
    row_edges, col_edges = _hqunet_grid_edges(G)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, params):
        for l in range(int(n_layers)):
            for q in range(n_qubits):
                qml.RX(x[q, 0], wires=q)
                qml.RY(x[q, 1], wires=q)
                qml.RZ(x[q, 2], wires=q)
            for q1, q2 in row_edges:
                qml.RY(params[l, 0, 0], wires=q1)
                qml.RY(params[l, 0, 1], wires=q2)
                qml.CNOT(wires=[q1, q2])
                qml.RY(params[l, 0, 2], wires=q1)
                qml.RY(params[l, 0, 3], wires=q2)
            for q1, q2 in col_edges:
                qml.RY(params[l, 1, 0], wires=q1)
                qml.RY(params[l, 1, 1], wires=q2)
                qml.CNOT(wires=[q1, q2])
                qml.RY(params[l, 1, 2], wires=q1)
                qml.RY(params[l, 1, 3], wires=q2)
        if return_state:
            return qml.state()
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


def eval_hqunet_dual_expvals(x, params, grid: int = 2, n_layers: int = 2) -> np.ndarray:
    """Evaluate dual-basis [<Z_0..Nq-1>, <X_0..Nq-1>] expectation values for HQ-UNet."""
    G = int(grid)
    n_qubits = G * G
    state_qnode = build_hqunet_circuit(grid=G, n_layers=n_layers, return_state=True)
    state = np.asarray(state_qnode(x, params), dtype=np.complex128).reshape(-1)
    exp_z = []
    exp_x = []
    for q in range(n_qubits):
        s_q = state.reshape(1 << q, 2, 1 << (n_qubits - 1 - q))
        s0 = s_q[:, 0, :]
        s1 = s_q[:, 1, :]
        p0 = np.abs(s0) ** 2
        p1 = np.abs(s1) ** 2
        exp_z.append(float((p0 - p1).sum()))
        exp_x.append(float((2.0 * (s0.real * s1.real + s0.imag * s1.imag)).sum()))
    return np.array(exp_z + exp_x, dtype=float)


def _sample_inputs(seed: int = 42, ansatz: str = "legacy", n_layers: int = 2, hqunet_grid: int = 2):
    """Reasonable example values to show a non-trivial circuit evaluation.

    Inputs (x) match the range of spatial averages after upstream
    normalization/activations — arbitrary small angles are fine for display.
    Params follow the initialization used in QuantumBlock: small Gaussian.
    """
    rng = np.random.default_rng(seed)
    if ansatz == "hqunet":
        n_qubits = int(hqunet_grid) ** 2
        x = np.pi * np.tanh(rng.uniform(low=-1.0, high=1.0, size=(n_qubits, 3)))
        params = rng.uniform(low=-np.pi, high=np.pi, size=(n_layers, 2, 4))
        return x.astype(float), params.astype(float)
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


def save_expval_barplot(circuit, x, params, path, ansatz: str = "reupload", hqunet_grid: int = 2, n_layers: int = 2):
    """Bar chart of the per-qubit expectation output for the sample input."""
    if ansatz == "hqunet":
        vals = eval_hqunet_dual_expvals(x, params, grid=hqunet_grid, n_layers=n_layers)
        n_q = int(hqunet_grid) ** 2
        labels = [f"q{i}\n<Z>" for i in range(n_q)] + [f"q{i}\n<X>" for i in range(n_q)]
        palette = ["#4C72B0"] * n_q + ["#55A868"] * n_q
        fig_w = max(7.0, 0.55 * len(labels))
    else:
        vals = np.array(circuit(x, params), dtype=float)
        labels = [f"q{i}\n<Z>" for i in range(N_QUBITS)]
        palette = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"][:N_QUBITS]
        fig_w = 6.0
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    bars = ax.bar(
        labels,
        vals,
        color=palette,
        edgecolor="white",
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.02 if v >= 0 else -0.05),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8 if len(labels) > 8 else 9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Expectation value")
    x_summary = f"x={np.round(x, 3).tolist()}" if np.ndim(x) == 1 else f"x: {np.shape(x)} angles"
    ax.set_title(
        "Sample output of the quantum circuit\n"
        f"{x_summary}"
        + (f"   θ={np.round(params, 3).tolist()}" if np.ndim(params) == 1 else f"   θ: {np.shape(params)} angles"),
        fontsize=10,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return vals


def _draw_gate_box(ax, x_center, y_center, width, height, label, facecolor, edgecolor, fontsize=9.5):
    box = FancyBboxPatch(
        (x_center - width / 2, y_center - height / 2),
        width,
        height,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x_center, y_center, label, ha="center", va="center", fontsize=fontsize, zorder=4)


def _draw_cnot(ax, x_pos, y_ctrl, y_targ):
    ax.plot([x_pos, x_pos], [y_ctrl, y_targ], color="#111111", linewidth=1.3, zorder=3)
    ctrl = Circle((x_pos, y_ctrl), radius=0.08, facecolor="#111111", edgecolor="#111111", zorder=5)
    ax.add_patch(ctrl)
    targ = Circle((x_pos, y_targ), radius=0.17, facecolor="white", edgecolor="#111111", linewidth=1.3, zorder=4)
    ax.add_patch(targ)
    ax.plot([x_pos - 0.17, x_pos + 0.17], [y_targ, y_targ], color="#111111", linewidth=1.2, zorder=5)
    ax.plot([x_pos, x_pos], [y_targ - 0.17, y_targ + 0.17], color="#111111", linewidth=1.2, zorder=5)


def _draw_meter(ax, x_center, y_center, width=0.72, height=0.52):
    rect = mpatches.Rectangle(
        (x_center - width / 2, y_center - height / 2),
        width,
        height,
        facecolor="#FFF4D6",
        edgecolor="#B5851D",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(rect)
    arc = Arc(
        (x_center, y_center - 0.10),
        width=width * 0.65,
        height=height * 0.72,
        angle=0,
        theta1=15,
        theta2=165,
        color="#B5851D",
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(arc)
    ax.plot(
        [x_center, x_center + width * 0.20],
        [y_center - 0.12, y_center + 0.16],
        color="#B5851D",
        linewidth=1.2,
        zorder=4,
    )


def save_comparison_diagram(out_dir: str, n_layers: int = 2):
    """Render the two-panel legacy vs. reupload circuit schematic to PNG and PDF."""
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "quaffle_circuits_legacy_vs_reupload.png")
    pdf_path = os.path.join(out_dir, "quaffle_circuits_legacy_vs_reupload.pdf")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13.5, 9.2))
    wire_ys = [3.0, 2.0, 1.0, 0.0]

    # ── Top panel: Legacy circuit ────────────────────────────────────────────
    ax_top.set_xlim(-0.8, 12.2)
    ax_top.set_ylim(-1.15, 4.15)
    ax_top.axis("off")
    ax_top.text(
        -0.6,
        3.95,
        "Original circuit (q_ansatz='legacy', the paper's Figure 5)   |   1 layer, 4 trainable parameters per circuit",
        fontsize=11.5,
        ha="left",
        va="center",
    )

    for i, y in enumerate(wire_ys):
        ax_top.text(-0.15, y, rf"$q_{i}\;|0\rangle$", ha="right", va="center", fontsize=10.5)
        ax_top.plot([0.0, 7.6], [y, y], color="#333333", linewidth=0.9, zorder=1)
        _draw_gate_box(ax_top, 0.85, y, 1.15, 0.56, rf"$R_X(z_{i})$", "#D9E8FB", "#2F5597")
        _draw_gate_box(ax_top, 2.35, y, 1.15, 0.56, rf"$R_Y(\varphi_{i})$", "#DCF2DE", "#2E7D32")
        _draw_meter(ax_top, 5.95, y, width=0.80, height=0.56)
        ax_top.text(6.55, y, rf"$\langle Z_{i}\rangle$", ha="left", va="center", fontsize=10.0)

    cnot_xs_legacy = [3.45, 3.95, 4.45, 4.95]
    for step_idx, cx in enumerate(cnot_xs_legacy):
        y_c = wire_ys[step_idx]
        y_t = wire_ys[(step_idx + 1) % N_QUBITS]
        _draw_cnot(ax_top, cx, y_c, y_t)

    # Group dashed boxes on top panel
    group_specs_top = [
        (0.15, 1.55, "encoding", "#2F5597"),
        (1.65, 3.05, "trainable", "#2E7D32"),
        (3.20, 5.20, "CNOT ring", "#333333"),
    ]
    for x0, x1, glabel, gcol in group_specs_top:
        rect = mpatches.Rectangle(
            (x0, -0.45),
            x1 - x0,
            4.05,
            fill=False,
            edgecolor=gcol,
            linestyle="--",
            linewidth=1.0,
            zorder=2,
        )
        ax_top.add_patch(rect)
        ax_top.text((x0 + x1) / 2, 3.72, glabel, color=gcol, fontsize=9.5, ha="center", va="bottom")

    ax_top.text(
        0.15,
        -0.82,
        r"Outputs reduce to products of $c_i = \cos(z_i)\cos(\varphi_i)$:  "
        r"$\langle Z_0\rangle = c_1 c_2 c_3$,  "
        r"$\langle Z_1\rangle = c_0 c_1$,  "
        r"$\langle Z_2\rangle = c_0 c_1 c_2$,  "
        r"$\langle Z_3\rangle = c_0 c_1 c_2 c_3$",
        color="#9B1B1B",
        fontsize=9.8,
        ha="left",
        va="center",
    )

    # ── Bottom panel: Data-reuploading circuit ───────────────────────────────
    n_layers = max(1, int(n_layers))
    layer_width = 5.05
    total_x = 0.25 + n_layers * layer_width + 1.8
    ax_bot.set_xlim(-0.8, max(12.2, total_x))
    ax_bot.set_ylim(-1.15, 4.15)
    ax_bot.axis("off")
    ax_bot.text(
        -0.6,
        3.95,
        f"New circuit (q_ansatz='reupload', current default)   |   "
        f"{n_layers} layers, {16 * n_layers} trainable parameters per circuit",
        fontsize=11.5,
        ha="left",
        va="center",
    )

    meter_x = 0.25 + n_layers * layer_width + 0.55
    wire_end_x = meter_x + 1.15
    for i, y in enumerate(wire_ys):
        ax_bot.text(-0.15, y, rf"$q_{i}\;|0\rangle$", ha="right", va="center", fontsize=10.5)
        ax_bot.plot([0.0, wire_end_x], [y, y], color="#333333", linewidth=0.9, zorder=1)
        _draw_meter(ax_bot, meter_x, y, width=0.68, height=0.54)
        ax_bot.text(meter_x + 0.48, y, rf"$\langle Z_{i}\rangle$", ha="left", va="center", fontsize=10.0)

    for l_idx in range(n_layers):
        l_num = l_idx + 1
        x_base = 0.20 + l_idx * layer_width
        enc_x = x_base + 0.85
        rot_x = x_base + 2.55
        cnot_xs = [x_base + 3.60 + 0.36 * k for k in range(N_QUBITS)]

        for i, y in enumerate(wire_ys):
            _draw_gate_box(
                ax_bot,
                enc_x,
                y,
                1.52,
                0.56,
                rf"$R_Y(\pi\tanh(s_{i}^{{({l_num})}} z_{i}))$",
                "#D9E8FB",
                "#2F5597",
                fontsize=8.8,
            )
            _draw_gate_box(
                ax_bot,
                rot_x,
                y,
                1.52,
                0.56,
                rf"$\mathrm{{Rot}}(\phi, \theta, \omega)_{i}^{{({l_num})}}$",
                "#DCF2DE",
                "#2E7D32",
                fontsize=8.8,
            )

        for step_idx, cx in enumerate(cnot_xs):
            y_c = wire_ys[step_idx]
            y_t = wire_ys[(step_idx + 1) % N_QUBITS]
            _draw_cnot(ax_bot, cx, y_c, y_t)

        layer_rect = mpatches.Rectangle(
            (x_base, -0.45),
            layer_width - 0.10,
            4.05,
            fill=False,
            edgecolor="#6A3D9A",
            linestyle="--",
            linewidth=1.0,
            zorder=2,
        )
        ax_bot.add_patch(layer_rect)
        ax_bot.text(
            x_base + (layer_width - 0.10) / 2,
            3.72,
            f"layer {l_num}: re-encode input, trainable rotations, CNOT ring",
            color="#6A3D9A",
            fontsize=9.3,
            ha="center",
            va="bottom",
        )

    ax_bot.text(
        0.20,
        -0.78,
        r"$s =$ trainable input scale per qubit and layer.  "
        r"$\mathrm{Rot} =$ general single-qubit rotation (3 trainable angles).  "
        r"Same input $z$ is encoded again in each layer (data re-uploading).",
        color="#222222",
        fontsize=9.5,
        ha="left",
        va="center",
    )

    fig.text(
        0.02,
        0.015,
        "Both: one circuit per group of 4 bottleneck channels (8 groups for 32 quantum channels). "
        "Input z = each channel averaged over the whole image. Output: 4 expectation values per group.",
        fontsize=9.2,
        color="#333333",
    )

    plt.tight_layout(rect=[0.0, 0.035, 1.0, 1.0])
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Visualize the QUAFFLE quantum circuit.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for sample input/params (default: 42)")
    parser.add_argument("--out-dir", "--out", dest="out_dir", default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--q-ansatz", choices=["reupload", "legacy", "hqunet", "compare"], default="reupload",
                        help="Circuit to draw: reupload, legacy, hqunet, or compare (default: reupload)")
    parser.add_argument("--compare-legacy-reupload", action="store_true",
                        help="Alias for --q-ansatz compare")
    parser.add_argument("--q-layers", type=int, default=2,
                        help="Circuit layers for --q-ansatz reupload/hqunet/compare (default: 2)")
    parser.add_argument("--hqunet-grid", type=int, choices=[2, 4], default=2,
                        help="2D patch grid size G in {2, 4} for --q-ansatz hqunet (default: 2 -> 4 qubits)")
    args = parser.parse_args(argv)
    if args.compare_legacy_reupload:
        args.q_ansatz = "compare"
    return args


def main(argv=None):
    args = parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.q_ansatz == "compare":
        png_path, pdf_path = save_comparison_diagram(args.out_dir, n_layers=args.q_layers)
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
        return png_path, pdf_path

    if args.q_ansatz == "reupload":
        circuit = build_reupload_circuit(N_QUBITS, args.q_layers)
        suffix = f"_reupload_L{args.q_layers}"
    elif args.q_ansatz == "hqunet":
        circuit = build_hqunet_circuit(grid=args.hqunet_grid, n_layers=args.q_layers)
        suffix = f"_hqunet_G{args.hqunet_grid}_L{args.q_layers}"
    else:
        circuit = build_circuit(N_QUBITS)
        suffix = ""
    x, params = _sample_inputs(
        seed=args.seed,
        ansatz=args.q_ansatz,
        n_layers=args.q_layers,
        hqunet_grid=args.hqunet_grid,
    )

    txt_path = os.path.join(args.out_dir, f"quantum_circuit{suffix}.txt")
    png_path = os.path.join(args.out_dir, f"quantum_circuit{suffix}.png")
    bar_path = os.path.join(args.out_dir, f"quantum_circuit{suffix}_expvals.png")

    text = save_text_diagram(circuit, x, params, txt_path)
    save_mpl_diagram(circuit, x, params, png_path)
    vals = save_expval_barplot(
        circuit,
        x,
        params,
        bar_path,
        ansatz=args.q_ansatz,
        hqunet_grid=args.hqunet_grid,
        n_layers=args.q_layers,
    )

    print(text)
    print()
    print(f"Saved: {txt_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {bar_path}")
    print(f"Sample outputs: {np.round(vals, 4).tolist()}")


if __name__ == "__main__":
    main()
