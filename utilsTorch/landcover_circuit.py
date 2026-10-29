"""Isolated one-layer, 16-qubit circuit for the native LandCover.ai pilot.

Historical HQ source bytes and sequential behavior remain unchanged. This
module reuses their elementary gate matrices/readout, but owns the noncommuting
brickwork schedule in HQ-UNet Fig. 2. The sequential option exists for independent
parity checks against the historical simulator; the native model uses brickwork.
The model evaluates one sample at a time with activation checkpointing on CPU.
"""
import torch

from .unetTorch import (
    _dual_basis_expvals_from_state, _hqunet_filter_matrix, _rx_mat, _ry_mat, _rz_mat,
)


def native_hqunet_expvals(angles, weights, *, edge_order='brickwork'):
    """Return float64 [B,32] Z-then-X expectations, without changing angle units.

    ``angles`` is floating [B,16,3] RX/RY/RZ radians and ``weights`` is exactly
    floating [1,2,4]: one shared horizontal and one shared vertical filter. Each
    filter is RY(theta1/theta2), CNOT(low→high), RY(phi1/phi2). Full statevectors
    use complex128. Direct batches are bounded at eight; the memory-saving
    model wrapper always calls with B=1. No shot noise or quantum pooling is used.
    """
    if (not torch.is_tensor(angles) or not torch.is_tensor(weights) or
            not angles.is_floating_point() or not weights.is_floating_point() or
            angles.ndim != 3 or not 1 <= angles.shape[0] <= 8 or tuple(angles.shape[1:]) != (16,3) or
            tuple(weights.shape) != (1,2,4)):
        raise ValueError('Expected floating angles [B,16,3], 1<=B<=8, and shared weights [1,2,4]')
    if angles.device != weights.device:
        raise ValueError('Angles and weights must use the same device')
    if not bool(torch.isfinite(angles).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError('Circuit angles and weights must be finite')
    if edge_order == 'brickwork':
        row_edges = [(4*r+c,4*r+c+1) for r in range(4) for parity in (0,1) for c in range(parity,3,2)]
        col_edges = [(4*r+c,4*(r+1)+c) for c in range(4) for parity in (0,1) for r in range(parity,3,2)]
    elif edge_order == 'sequential':
        row_edges = [(4*r+c,4*r+c+1) for r in range(4) for c in range(3)]
        col_edges = [(4*r+c,4*(r+1)+c) for r in range(3) for c in range(4)]
    else:
        raise ValueError("edge_order must be 'brickwork' or 'sequential'")
    batch = angles.shape[0]
    angles64,weights64 = angles.to(torch.float64),weights.to(torch.float64)
    state = torch.zeros((batch,1 << 16),dtype=torch.complex128,device=angles.device)
    state[:,0] = 1.+0.j

    def apply_two(st,first,second,gate):
        a_len,m_len,c_len = 1 << first,1 << (second-first-1),1 << (15-second)
        pair = st.reshape(batch,a_len,2,m_len,2,c_len)
        s00,s01 = pair[:,:,0,:,0,:],pair[:,:,0,:,1,:]
        s10,s11 = pair[:,:,1,:,0,:],pair[:,:,1,:,1,:]
        vector = torch.stack([s00,s01,s10,s11],dim=-1)
        out = torch.matmul(vector,gate.t())
        row0 = torch.stack([out[...,0],out[...,1]],dim=3)
        row1 = torch.stack([out[...,2],out[...,3]],dim=3)
        return torch.stack([row0,row1],dim=2).reshape(batch,1 << 16)

    encoding = _rz_mat(angles64[:,:,2]) @ _ry_mat(angles64[:,:,1]) @ _rx_mat(angles64[:,:,0])
    for qubit in range(16):
        part = state.reshape(batch,1 << qubit,2,1 << (15-qubit))
        s0,s1 = part[:,:,0,:],part[:,:,1,:]
        gate = encoding[:,qubit]
        g00,g01 = gate[:,0,0].view(batch,1,1),gate[:,0,1].view(batch,1,1)
        g10,g11 = gate[:,1,0].view(batch,1,1),gate[:,1,1].view(batch,1,1)
        state = torch.stack([g00*s0+g01*s1,g10*s0+g11*s1],dim=2).reshape(batch,1 << 16)
    for direction,edges in enumerate((row_edges,col_edges)):
        values = weights64[0,direction]
        real_gate = _hqunet_filter_matrix(values[0],values[1],values[2],values[3])
        gate = torch.complex(real_gate,torch.zeros_like(real_gate))
        for first,second in edges:
            state = apply_two(state,first,second,gate)
    return _dual_basis_expvals_from_state(state,16)
