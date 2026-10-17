# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from outlier_energy_utils import compute_leave_one_out_energy, compute_outlier_energy, normalize_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Sinkhorn-outlier-energy (SOE) latent outliers with semantic anchoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--token_embed_path", type=str, default="./token_embed_in100.npy")
    parser.add_argument("--id_feat_path", type=str, default="./id_feat_in100_99epoch.npy")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--energy_eps", type=float, default=0.05)
    parser.add_argument("--gaussian_mag_ood_det", type=float, default=0.012)
    parser.add_argument("--anchor_topk", type=int, default=200)
    parser.add_argument("--anchor_pool_size", type=int, default=400)
    parser.add_argument("--anchor_semantic_quantile", type=float, default=0.25)
    parser.add_argument("--token_cos_floor_min", type=float, default=0.25)
    parser.add_argument("--token_cos_floor_max", type=float, default=1.0)
    parser.add_argument("--seed_cos_floor", type=float, default=0.70)
    parser.add_argument("--local_k", type=int, default=128)
    parser.add_argument("--num_samples_per_class", type=int, default=10000)
    parser.add_argument("--proposals_per_anchor", type=int, default=256)
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--anchor_batch_size", type=int, default=32)
    parser.add_argument("--candidate_chunk_size", type=int, default=4096)
    parser.add_argument("--reference_chunk_size", type=int, default=256)
    parser.add_argument("--threshold_margin", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_classes", type=int, default=0,
                        help="Optional smoke-test limit; 0 means use all classes.")
    parser.add_argument("--log_every", type=int, default=5)
    return parser.parse_args()


def resolve_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_output_name(args):
    name = (
        "outlier_energy_semantic_embed"
        + "_eps_" + str(args.energy_eps)
        + "_noise_" + str(args.gaussian_mag_ood_det)
        + "_anchor_" + str(args.anchor_topk)
        + "_pool_" + str(args.anchor_pool_size)
        + "_samples_" + str(args.num_samples_per_class)
        + "_prop_" + str(args.proposals_per_anchor)
        + "_localk_" + str(args.local_k)
        + "_tcfloor_" + str(args.token_cos_floor_min)
        + "_seedcos_" + str(args.seed_cos_floor)
    )
    if args.token_cos_floor_max < 0.999999:
        name += "_tcfmax_" + str(args.token_cos_floor_max)
    return name + ".npy"


@torch.no_grad()
def compute_local_energy_batched(proposals, local_reference, energy_eps):
    if proposals.ndim != 3:
        raise ValueError("proposals must be a 3D tensor")
    if local_reference.ndim != 3:
        raise ValueError("local_reference must be a 3D tensor")
    if proposals.size(0) != local_reference.size(0):
        raise ValueError("batch size mismatch between proposals and local_reference")
    if proposals.size(2) != local_reference.size(2):
        raise ValueError("feature dim mismatch between proposals and local_reference")
    if local_reference.size(1) == 0:
        raise ValueError("local_reference must contain at least one point")

    log_norm = math.log(local_reference.size(1))
    sim = torch.matmul(proposals, local_reference.transpose(1, 2))
    log_mass = torch.logsumexp((sim - 1.0) / energy_eps, dim=2) - log_norm
    return -energy_eps * log_mass


@torch.no_grad()
def build_local_reference(reference, anchors, local_k):
    neighborhood_size = min(local_k, reference.size(0))
    similarity = anchors @ reference.t()
    neighbor_indices = torch.topk(similarity, k=neighborhood_size, dim=1, largest=True).indices
    return reference[neighbor_indices]


@torch.no_grad()
def select_anchor_indices(reference_energy, reference_token_cos, args):
    anchor_count = min(args.anchor_topk, reference_energy.size(0))
    pool_size = min(args.anchor_pool_size, reference_energy.size(0))
    pool_indices = torch.topk(reference_energy, k=pool_size, largest=True).indices
    pool_token_cos = reference_token_cos[pool_indices]

    pool_floor = max(args.token_cos_floor_min, torch.quantile(pool_token_cos, args.anchor_semantic_quantile).item())
    pool_floor = min(pool_floor, args.token_cos_floor_max)
    semantic_pool_indices = pool_indices[pool_token_cos >= pool_floor]

    anchor_selection_fallback = 0
    if semantic_pool_indices.numel() < anchor_count:
        anchor_indices = torch.topk(reference_energy, k=anchor_count, largest=True).indices
        anchor_selection_fallback = 1
    else:
        semantic_pool_values = reference_energy[semantic_pool_indices]
        keep = torch.topk(semantic_pool_values, k=anchor_count, largest=True).indices
        anchor_indices = semantic_pool_indices[keep]

    anchor_values = reference_energy[anchor_indices]
    selected_anchor_cos = reference_token_cos[anchor_indices]
    token_cos_floor = max(args.token_cos_floor_min, torch.quantile(selected_anchor_cos, args.anchor_semantic_quantile).item())
    token_cos_floor = min(token_cos_floor, args.token_cos_floor_max)
    return anchor_values, anchor_indices, token_cos_floor, {
        "anchor_count": int(anchor_count),
        "anchor_pool_size": int(pool_size),
        "anchor_selection_fallback": int(anchor_selection_fallback),
    }


@torch.no_grad()
def sample_anchor_chunk(anchors, reference, local_reference, token_anchor, token_cos_floor, threshold, keep_per_anchor, args):
    if keep_per_anchor > args.proposals_per_anchor * args.max_rounds:
        raise ValueError(
            "keep_per_anchor={} exceeds proposals_per_anchor*max_rounds={}".format(
                keep_per_anchor, args.proposals_per_anchor * args.max_rounds
            )
        )

    candidate_rounds = []
    rank_rounds = []
    global_energy_rounds = []
    local_energy_rounds = []
    token_cos_rounds = []
    seed_cos_rounds = []
    strong_mask_rounds = []
    token_seed_mask_rounds = []
    token_only_mask_rounds = []
    total_rounds = 0

    for round_index in range(args.max_rounds):
        total_rounds = round_index + 1
        noise = torch.randn(
            anchors.size(0),
            args.proposals_per_anchor,
            anchors.size(1),
            device=anchors.device,
            dtype=anchors.dtype,
        )
        proposals = normalize_features(anchors.unsqueeze(1) + args.gaussian_mag_ood_det * noise)
        global_energies = compute_outlier_energy(
            proposals.reshape(-1, anchors.size(1)),
            reference,
            energy_eps=args.energy_eps,
            candidate_chunk_size=args.candidate_chunk_size,
        ).view(anchors.size(0), args.proposals_per_anchor)
        local_energies = compute_local_energy_batched(proposals, local_reference, args.energy_eps)
        token_cos = torch.sum(proposals * token_anchor.view(1, 1, -1), dim=2)
        seed_cos = torch.sum(proposals * anchors.unsqueeze(1), dim=2)

        strong_mask = (global_energies >= threshold) & (token_cos >= token_cos_floor) & (seed_cos >= args.seed_cos_floor)
        token_seed_mask = (token_cos >= token_cos_floor) & (seed_cos >= args.seed_cos_floor)
        token_only_mask = token_cos >= token_cos_floor

        candidate_rounds.append(proposals)
        rank_rounds.append(local_energies)
        global_energy_rounds.append(global_energies)
        local_energy_rounds.append(local_energies)
        token_cos_rounds.append(token_cos)
        seed_cos_rounds.append(seed_cos)
        strong_mask_rounds.append(strong_mask)
        token_seed_mask_rounds.append(token_seed_mask)
        token_only_mask_rounds.append(token_only_mask)

        cumulative_strong = torch.cat(strong_mask_rounds, dim=1)
        if torch.all(cumulative_strong.sum(dim=1) >= keep_per_anchor):
            break

    all_candidates = torch.cat(candidate_rounds, dim=1)
    all_rank = torch.cat(rank_rounds, dim=1)
    all_global = torch.cat(global_energy_rounds, dim=1)
    all_local = torch.cat(local_energy_rounds, dim=1)
    all_token = torch.cat(token_cos_rounds, dim=1)
    all_seed = torch.cat(seed_cos_rounds, dim=1)
    all_strong = torch.cat(strong_mask_rounds, dim=1)
    all_token_seed = torch.cat(token_seed_mask_rounds, dim=1)
    all_token_only = torch.cat(token_only_mask_rounds, dim=1)

    selected_candidates = []
    selected_rank = []
    selected_global = []
    selected_local = []
    selected_token = []
    selected_seed = []

    threshold_fallback = 0
    semantic_only_fallback = 0
    token_only_fallback = 0
    all_candidate_fallback = 0
    feasible_fill = 0.0

    for anchor_index in range(anchors.size(0)):
        strong_mask = all_strong[anchor_index]
        token_seed_mask = all_token_seed[anchor_index]
        token_only_mask = all_token_only[anchor_index]

        strong_count = int(strong_mask.sum().item())
        feasible_fill += min(strong_count, keep_per_anchor) / float(keep_per_anchor)

        if strong_count >= keep_per_anchor:
            source_mask = strong_mask
        elif int(token_seed_mask.sum().item()) >= keep_per_anchor:
            source_mask = token_seed_mask
            threshold_fallback += 1
        elif int(token_only_mask.sum().item()) >= keep_per_anchor:
            source_mask = token_only_mask
            semantic_only_fallback += 1
        else:
            source_mask = torch.ones_like(strong_mask, dtype=torch.bool)
            token_only_fallback += 1
            all_candidate_fallback += 1

        source_rank = all_rank[anchor_index][source_mask]
        topk = torch.topk(source_rank, k=keep_per_anchor, largest=True)
        chosen = topk.indices

        selected_candidates.append(all_candidates[anchor_index][source_mask][chosen])
        selected_rank.append(source_rank[chosen])
        selected_global.append(all_global[anchor_index][source_mask][chosen])
        selected_local.append(all_local[anchor_index][source_mask][chosen])
        selected_token.append(all_token[anchor_index][source_mask][chosen])
        selected_seed.append(all_seed[anchor_index][source_mask][chosen])

    return (
        torch.stack(selected_candidates, dim=0),
        torch.stack(selected_rank, dim=0),
        torch.stack(selected_global, dim=0),
        torch.stack(selected_local, dim=0),
        torch.stack(selected_token, dim=0),
        torch.stack(selected_seed, dim=0),
        {
            "threshold_fallback_batches": int(threshold_fallback),
            "semantic_only_fallback_batches": int(semantic_only_fallback),
            "token_only_fallback_batches": int(token_only_fallback),
            "all_candidate_fallback_batches": int(all_candidate_fallback),
            "avg_feasible_fill_ratio": float(feasible_fill / max(anchors.size(0), 1)),
            "rounds_used": int(total_rounds),
        },
    )


@torch.no_grad()
def generate_class_outliers(reference_features, anchor_vector, args):
    reference = normalize_features(reference_features)
    anchor_norm = anchor_vector.norm()
    token_anchor = normalize_features(anchor_vector.unsqueeze(0)).squeeze(0)

    reference_energy = compute_leave_one_out_energy(
        reference,
        energy_eps=args.energy_eps,
        reference_chunk_size=args.reference_chunk_size,
    )
    reference_token_cos = reference @ token_anchor

    anchor_values, anchor_indices, token_cos_floor, anchor_stats = select_anchor_indices(
        reference_energy,
        reference_token_cos,
        args,
    )
    threshold = anchor_values.min() + args.threshold_margin
    keep_per_anchor = int(math.ceil(float(args.num_samples_per_class) / float(anchor_stats["anchor_count"])))

    class_candidates = []
    class_rank = []
    class_global_energy = []
    class_local_energy = []
    class_token_cos = []
    class_seed_cos = []

    total_threshold_fallback = 0
    total_semantic_only_fallback = 0
    total_token_only_fallback = 0
    total_all_candidate_fallback = 0
    total_rounds = 0
    feasible_fraction_total = 0.0
    feasible_fraction_count = 0

    for start in range(0, anchor_stats["anchor_count"], args.anchor_batch_size):
        end = min(start + args.anchor_batch_size, anchor_stats["anchor_count"])
        anchor_chunk = reference[anchor_indices[start:end]]
        local_reference = build_local_reference(reference, anchor_chunk, args.local_k)
        selected = sample_anchor_chunk(
            anchor_chunk,
            reference,
            local_reference,
            token_anchor,
            token_cos_floor,
            threshold,
            keep_per_anchor,
            args,
        )
        selected_chunk, selected_rank_chunk, selected_global_chunk, selected_local_chunk, selected_token_chunk, selected_seed_chunk, chunk_stats = selected
        class_candidates.append(selected_chunk.reshape(-1, reference.size(1)))
        class_rank.append(selected_rank_chunk.reshape(-1))
        class_global_energy.append(selected_global_chunk.reshape(-1))
        class_local_energy.append(selected_local_chunk.reshape(-1))
        class_token_cos.append(selected_token_chunk.reshape(-1))
        class_seed_cos.append(selected_seed_chunk.reshape(-1))

        total_threshold_fallback += chunk_stats["threshold_fallback_batches"]
        total_semantic_only_fallback += chunk_stats["semantic_only_fallback_batches"]
        total_token_only_fallback += chunk_stats["token_only_fallback_batches"]
        total_all_candidate_fallback += chunk_stats["all_candidate_fallback_batches"]
        total_rounds += chunk_stats["rounds_used"]
        feasible_fraction_total += chunk_stats["avg_feasible_fill_ratio"]
        feasible_fraction_count += 1

    class_candidates = torch.cat(class_candidates, dim=0)
    class_rank = torch.cat(class_rank, dim=0)
    class_global_energy = torch.cat(class_global_energy, dim=0)
    class_local_energy = torch.cat(class_local_energy, dim=0)
    class_token_cos = torch.cat(class_token_cos, dim=0)
    class_seed_cos = torch.cat(class_seed_cos, dim=0)

    if class_candidates.size(0) < args.num_samples_per_class:
        raise RuntimeError(
            "Only produced {} samples for class, expected {}".format(
                class_candidates.size(0), args.num_samples_per_class
            )
        )

    if class_candidates.size(0) > args.num_samples_per_class:
        keep = torch.topk(class_rank, k=args.num_samples_per_class, largest=True).indices
        class_candidates = class_candidates[keep]
        class_rank = class_rank[keep]
        class_global_energy = class_global_energy[keep]
        class_local_energy = class_local_energy[keep]
        class_token_cos = class_token_cos[keep]
        class_seed_cos = class_seed_cos[keep]

    return (
        class_candidates * anchor_norm,
        {
            "anchor_count": int(anchor_stats["anchor_count"]),
            "keep_per_anchor": int(keep_per_anchor),
            "threshold": float(threshold.item()),
            "token_cos_floor": float(token_cos_floor),
            "reference_energy_min": float(reference_energy.min().item()),
            "reference_energy_max": float(reference_energy.max().item()),
            "anchor_token_cos_mean": float(reference_token_cos[anchor_indices].mean().item()),
            "anchor_token_cos_min": float(reference_token_cos[anchor_indices].min().item()),
            "anchor_token_cos_max": float(reference_token_cos[anchor_indices].max().item()),
            "candidate_rank_min": float(class_rank.min().item()),
            "candidate_rank_max": float(class_rank.max().item()),
            "candidate_rank_mean": float(class_rank.mean().item()),
            "candidate_global_energy_min": float(class_global_energy.min().item()),
            "candidate_global_energy_max": float(class_global_energy.max().item()),
            "candidate_global_energy_mean": float(class_global_energy.mean().item()),
            "candidate_local_energy_min": float(class_local_energy.min().item()),
            "candidate_local_energy_max": float(class_local_energy.max().item()),
            "candidate_local_energy_mean": float(class_local_energy.mean().item()),
            "candidate_token_cos_min": float(class_token_cos.min().item()),
            "candidate_token_cos_max": float(class_token_cos.max().item()),
            "candidate_token_cos_mean": float(class_token_cos.mean().item()),
            "candidate_seed_cos_min": float(class_seed_cos.min().item()),
            "candidate_seed_cos_max": float(class_seed_cos.max().item()),
            "candidate_seed_cos_mean": float(class_seed_cos.mean().item()),
            "threshold_fallback_batches": int(total_threshold_fallback),
            "semantic_only_fallback_batches": int(total_semantic_only_fallback),
            "token_only_fallback_batches": int(total_token_only_fallback),
            "all_candidate_fallback_batches": int(total_all_candidate_fallback),
            "avg_feasible_fill_ratio": float(feasible_fraction_total / max(feasible_fraction_count, 1)),
            "rounds_used_total": int(total_rounds),
            "anchor_pool_size": int(anchor_stats["anchor_pool_size"]),
            "anchor_selection_fallback": int(anchor_stats["anchor_selection_fallback"]),
        },
    )


def main():
    args = parse_args()
    if args.energy_eps <= 0:
        raise ValueError("--energy_eps must be positive")
    if args.gaussian_mag_ood_det <= 0:
        raise ValueError("--gaussian_mag_ood_det must be positive")
    if args.anchor_topk <= 0:
        raise ValueError("--anchor_topk must be positive")
    if args.anchor_pool_size <= 0:
        raise ValueError("--anchor_pool_size must be positive")
    if not 0.0 <= args.anchor_semantic_quantile <= 1.0:
        raise ValueError("--anchor_semantic_quantile must be in [0, 1]")
    if args.token_cos_floor_max <= 0:
        raise ValueError("--token_cos_floor_max must be positive")
    if args.token_cos_floor_max < args.token_cos_floor_min:
        raise ValueError("--token_cos_floor_max must be >= --token_cos_floor_min")
    if args.num_samples_per_class <= 0:
        raise ValueError("--num_samples_per_class must be positive")
    if args.proposals_per_anchor <= 0:
        raise ValueError("--proposals_per_anchor must be positive")
    if args.max_rounds <= 0:
        raise ValueError("--max_rounds must be positive")
    if args.anchor_batch_size <= 0:
        raise ValueError("--anchor_batch_size must be positive")
    if args.candidate_chunk_size <= 0:
        raise ValueError("--candidate_chunk_size must be positive")
    if args.reference_chunk_size <= 0:
        raise ValueError("--reference_chunk_size must be positive")
    if args.local_k <= 0:
        raise ValueError("--local_k must be positive")
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    print("Compute device: {}".format(device), flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    anchor = torch.from_numpy(np.load(args.token_embed_path)).to(device=device, dtype=torch.float32)
    data_dict = torch.from_numpy(np.load(args.id_feat_path)).to(device=device, dtype=torch.float32)

    if data_dict.shape[0] != anchor.shape[0]:
        raise ValueError(
            "Feature file classes {} do not match token embeddings {}".format(
                data_dict.shape[0], anchor.shape[0]
            )
        )

    if args.max_classes > 0:
        limit = min(args.max_classes, anchor.shape[0])
        anchor = anchor[:limit]
        data_dict = data_dict[:limit]
        print("Using class limit {} for smoke run".format(limit), flush=True)

    output_samples = []
    class_stats = []
    start_time = time.time()

    for class_index in range(anchor.shape[0]):
        class_start = time.time()
        samples, stats = generate_class_outliers(data_dict[class_index], anchor[class_index], args)
        output_samples.append(samples)
        stats["class_index"] = int(class_index)
        class_stats.append(stats)

        if ((class_index + 1) % args.log_every == 0) or (class_index + 1 == anchor.shape[0]):
            print(
                "Class {} / {} done in {:.1f}s | thr {:.4f} | local {:.4f} | tok {:.4f} | fallbacks g/sem/tok/all = {}/{}/{}/{}".format(
                    class_index + 1,
                    anchor.shape[0],
                    time.time() - class_start,
                    stats["threshold"],
                    stats["candidate_local_energy_mean"],
                    stats["candidate_token_cos_mean"],
                    stats["threshold_fallback_batches"],
                    stats["semantic_only_fallback_batches"],
                    stats["token_only_fallback_batches"],
                    stats["all_candidate_fallback_batches"],
                ),
                flush=True,
            )

    output_array = torch.stack(output_samples, dim=0).cpu().numpy()
    output_name = build_output_name(args)
    output_path = os.path.join(args.output_dir, output_name)
    np.save(output_path, output_array)

    stats_path = output_path.replace(".npy", ".json")
    with open(stats_path, "w") as handle:
        json.dump(
            {
                "output_path": output_path,
                "shape": list(output_array.shape),
                "energy_eps": args.energy_eps,
                "gaussian_mag_ood_det": args.gaussian_mag_ood_det,
                "anchor_topk": args.anchor_topk,
                "anchor_pool_size": args.anchor_pool_size,
                "anchor_semantic_quantile": args.anchor_semantic_quantile,
                "token_cos_floor_min": args.token_cos_floor_min,
                "token_cos_floor_max": args.token_cos_floor_max,
                "seed_cos_floor": args.seed_cos_floor,
                "local_k": args.local_k,
                "num_samples_per_class": args.num_samples_per_class,
                "proposals_per_anchor": args.proposals_per_anchor,
                "max_rounds": args.max_rounds,
                "threshold_margin": args.threshold_margin,
                "device": str(device),
                "elapsed_seconds": time.time() - start_time,
                "class_stats": class_stats,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print("Saved embeddings: {}".format(output_path), flush=True)
    print("Saved summary: {}".format(stats_path), flush=True)


if __name__ == "__main__":
    main()
