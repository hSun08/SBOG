import math

import torch
import torch.nn.functional as F


def normalize_features(features):
    return F.normalize(features, p=2, dim=-1)


@torch.no_grad()
def compute_outlier_energy(candidates, reference, energy_eps, candidate_chunk_size=4096):
    if candidates.ndim != 2:
        raise ValueError("candidates must be a 2D tensor")
    if reference.ndim != 2:
        raise ValueError("reference must be a 2D tensor")
    if candidates.size(1) != reference.size(1):
        raise ValueError("candidate dim {} does not match reference dim {}".format(
            candidates.size(1), reference.size(1)
        ))
    if reference.size(0) == 0:
        raise ValueError("reference must contain at least one point")
    if energy_eps <= 0:
        raise ValueError("energy_eps must be positive")

    log_norm = math.log(reference.size(0))
    energies = torch.empty(candidates.size(0), device=candidates.device, dtype=candidates.dtype)
    for start in range(0, candidates.size(0), candidate_chunk_size):
        end = min(start + candidate_chunk_size, candidates.size(0))
        sim = candidates[start:end] @ reference.t()
        log_mass = torch.logsumexp((sim - 1.0) / energy_eps, dim=1) - log_norm
        energies[start:end] = -energy_eps * log_mass
    return energies


@torch.no_grad()
def compute_leave_one_out_energy(reference, energy_eps, reference_chunk_size=256):
    if reference.ndim != 2:
        raise ValueError("reference must be a 2D tensor")
    if reference.size(0) < 2:
        raise ValueError("reference must contain at least two points for leave-one-out energy")
    if energy_eps <= 0:
        raise ValueError("energy_eps must be positive")

    num_reference = reference.size(0)
    log_norm = math.log(num_reference - 1)
    energies = torch.empty(num_reference, device=reference.device, dtype=reference.dtype)

    for start in range(0, num_reference, reference_chunk_size):
        end = min(start + reference_chunk_size, num_reference)
        sim = reference[start:end] @ reference.t()
        row_index = torch.arange(end - start, device=reference.device)
        col_index = torch.arange(start, end, device=reference.device)
        sim[row_index, col_index] = float("-inf")
        log_mass = torch.logsumexp((sim - 1.0) / energy_eps, dim=1) - log_norm
        energies[start:end] = -energy_eps * log_mass

    return energies
