"""Backend dispatcher for the PyTorch3D APIs used in this repository.

Two backends are supported and chosen at import time via the
``GAUSSIAN_HS_USE_PYTORCH3D`` environment variable:

* unset / ``0`` (default) — use the in-house implementations defined below.
  These are pure-PyTorch equivalents of the upstream functions for the
  exact call patterns this repository uses (``knn_points`` with K=1,
  ``sample_farthest_points`` with ``random_start_point=False``,
  ``euler_angles_to_matrix`` with the ``XYZ`` convention). They have been
  numerically verified against the upstream reference.

* ``1`` — delegate directly to the installed ``pytorch3d`` package. Use
  this when you have a working pytorch3d build on the target machine
  (CUDA 12.8 / sm_120 / PyTorch 2.9) and want bit-identical behaviour to
  the original code.

See ``memo/pytorch3d_dual_mode.md`` for the rationale, demo-script
guidelines and how to scope the variable so it does not leak into the
caller's shell.
"""

import os
from functools import reduce

import torch


def _flag_truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


USE_PYTORCH3D: bool = _flag_truthy(os.environ.get("GAUSSIAN_HS_USE_PYTORCH3D", ""))

if USE_PYTORCH3D:
    try:
        from pytorch3d.ops import knn_points as _p3d_knn_points
        from pytorch3d.ops import sample_farthest_points as _p3d_sample_farthest_points
        from pytorch3d.transforms import euler_angles_to_matrix as _p3d_euler_angles_to_matrix
    except ImportError as exc:
        raise ImportError(
            "GAUSSIAN_HS_USE_PYTORCH3D=1 was set but the pytorch3d package "
            "could not be imported. Either install pytorch3d (see "
            "memo/environment_notes.md) or unset the variable to fall back "
            "to the in-house implementations."
        ) from exc


def use_pytorch3d() -> bool:
    """Whether the pytorch3d backend is currently selected."""
    return USE_PYTORCH3D


def knn_points(p1, p2, K=1, return_nn=False):
    if USE_PYTORCH3D:
        out = _p3d_knn_points(p1, p2, K=K, return_nn=return_nn)
        return out.dists, out.idx, out.knn

    if K < 1:
        raise ValueError("K must be >= 1")

    dist_chunks = []
    idx_chunks = []
    chunk_size = 4096
    for start in range(0, p1.shape[1], chunk_size):
        p1_chunk = p1[:, start:start + chunk_size]
        dists = torch.cdist(p1_chunk, p2, p=2) ** 2
        dists, idx = torch.topk(dists, k=K, dim=-1, largest=False, sorted=True)
        dist_chunks.append(dists)
        idx_chunks.append(idx)

    dists = torch.cat(dist_chunks, dim=1)
    idx = torch.cat(idx_chunks, dim=1)

    if not return_nn:
        return dists, idx, None

    idx_expand = idx.unsqueeze(-1).expand(*idx.shape, p2.shape[-1])
    p2_expand = p2.unsqueeze(1).expand(p1.shape[0], p1.shape[1], p2.shape[1], p2.shape[2])
    nn = torch.gather(p2_expand, 2, idx_expand)
    return dists, idx, nn


def sample_farthest_points(points, K):
    if USE_PYTORCH3D:
        return _p3d_sample_farthest_points(points, K=K)

    if points.ndim != 3:
        raise ValueError("points must have shape (N, P, D)")

    batch_size, num_points, dims = points.shape
    if isinstance(K, torch.Tensor):
        K = int(K.max().item())
    K = int(K)
    if K < 1:
        raise ValueError("K must be >= 1")

    sample_count = min(K, num_points)
    selected = torch.zeros(batch_size, K, dtype=torch.long, device=points.device)
    distances = torch.full((batch_size, num_points), float("inf"), dtype=points.dtype, device=points.device)
    farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
    batch_indices = torch.arange(batch_size, device=points.device)

    for i in range(sample_count):
        selected[:, i] = farthest
        centroid = points[batch_indices, farthest].view(batch_size, 1, dims)
        distances = torch.minimum(distances, ((points - centroid) ** 2).sum(-1))
        farthest = distances.max(-1).indices

    if sample_count < K:
        selected[:, sample_count:] = selected[:, sample_count - 1].unsqueeze(-1)

    sampled = torch.gather(points, 1, selected.unsqueeze(-1).expand(batch_size, K, dims))
    return sampled, selected


def _axis_angle_rotation(axis, angle):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention):
    if USE_PYTORCH3D:
        return _p3d_euler_angles_to_matrix(euler_angles, convention)

    if euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3 or len(set(convention)) != 3:
        raise ValueError("Convention must have 3 different letters.")
    if any(axis not in "XYZ" for axis in convention):
        raise ValueError(f"Invalid convention: {convention}")

    matrices = [
        _axis_angle_rotation(axis, angle)
        for axis, angle in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return reduce(torch.matmul, matrices)
