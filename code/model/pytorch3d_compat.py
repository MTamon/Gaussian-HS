from functools import reduce

import torch


def knn_points(p1, p2, K=1, return_nn=False):
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
