import argparse
import json
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from torch.utils.data import DataLoader, Subset

from resnet import ResNet_Model
from utils.display_results import get_measures
import utils.svhn_loader as svhn
from dataset_paths import (
    default_cifar100_data_candidates,
    default_cifar100_ood_candidates,
    format_missing_path_message,
    remove_data_parallel_prefix,
    resolve_path,
)


def build_detector(detector_variant):
    if detector_variant == 'imagenet':
        return torch.nn.Sequential(
            torch.nn.Linear(1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
        )
    return torch.nn.Sequential(torch.nn.Linear(1, 2))


def infer_bundle_path(load_path):
    if not load_path:
        return ''
    root, ext = os.path.splitext(load_path)
    candidate = root + '_bundle' + ext
    if os.path.isfile(candidate):
        return candidate
    return ''


def extract_net_state(checkpoint):
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        return checkpoint['net']
    return checkpoint


def load_detector_bundle(load_path):
    if not load_path:
        return None
    bundle = torch.load(load_path, map_location='cpu')
    if not isinstance(bundle, dict) or 'detector' not in bundle:
        raise ValueError('Detector bundle {} does not contain detector state'.format(load_path))
    return bundle


def format_triplet(triplet):
    return '{:.2f}+-{:.2f}'.format(triplet['mean'], triplet['std'])


def percentize(point, values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'point': 100.0 * float(point),
        'mean': 100.0 * float(values.mean()),
        'std': 100.0 * float(values.std()),
    }


def bootstrap_from_scores(pos_scores, neg_scores, trials, seed):
    rng = np.random.default_rng(seed)
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)
    pos_n = len(pos_scores)
    neg_n = len(neg_scores)
    aurocs, auprs, fprs = [], [], []
    for _ in range(trials):
        pos_bs = pos_scores[rng.integers(0, pos_n, size=pos_n)]
        neg_bs = neg_scores[rng.integers(0, neg_n, size=neg_n)]
        auroc, aupr, fpr = get_measures(pos_bs, neg_bs)
        aurocs.append(auroc)
        auprs.append(aupr)
        fprs.append(fpr)
    return np.asarray(aurocs), np.asarray(auprs), np.asarray(fprs)


parser = argparse.ArgumentParser(description='Bootstrap evaluation for CIFAR100 OOD detection')
parser.add_argument('--method_name', type=str, required=True)
parser.add_argument('--load', type=str, required=True)
parser.add_argument('--detector_load', type=str, default='')
parser.add_argument('--detector_variant', choices=['default', 'imagenet'], default='imagenet')
parser.add_argument('--score', default='energy', choices=['MSP', 'energy', 'detector'])
parser.add_argument('--T', default=1.0, type=float)
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--prefetch', type=int, default=4)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--download', action='store_true')
parser.add_argument('--textures_root', type=str, default='')
parser.add_argument('--svhn_root', type=str, default='')
parser.add_argument('--places_root', type=str, default='')
parser.add_argument('--lsun_root', type=str, default='')
parser.add_argument('--isun_root', type=str, default='')
parser.add_argument('--bootstrap_trials', type=int, default=500)
parser.add_argument('--bootstrap_seed', type=int, default=1)
parser.add_argument('--subset_seed', type=int, default=1)
parser.add_argument('--ood_num_examples', type=int, default=0,
                    help='Number of OOD examples per dataset. Defaults to len(test_data)//5 to match the original step5.')
parser.add_argument('--output_json', type=str, default='')
args = parser.parse_args()
print(args, flush=True)

args.data_root, searched_data_roots = resolve_path(args.data_root, default_cifar100_data_candidates())
print('Resolved CIFAR-100 data_root:', args.data_root, flush=True)
if searched_data_roots:
    print('CIFAR-100 data root candidates:', searched_data_roots, flush=True)

ood_defaults = default_cifar100_ood_candidates()
resolved_roots = {}
missing = []
for flag, explicit in {
    'textures_root': args.textures_root,
    'svhn_root': args.svhn_root,
    'places_root': args.places_root,
    'lsun_root': args.lsun_root,
    'isun_root': args.isun_root,
}.items():
    resolved, searched = resolve_path(explicit, ood_defaults[flag])
    resolved_roots[flag] = resolved
    print('Resolved {}: {}'.format(flag, resolved), flush=True)
    if flag == 'svhn_root':
        if not args.download and (not resolved or not os.path.isdir(resolved)):
            missing.append(format_missing_path_message(flag, searched))
        continue
    if not resolved or not os.path.isdir(resolved):
        missing.append(format_missing_path_message(flag, searched))

if missing:
    raise FileNotFoundError('\n\n'.join(missing))

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

test_data = dset.CIFAR100(args.data_root, train=False, transform=test_transform, download=args.download)
num_classes = 100
test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                         num_workers=args.prefetch, pin_memory=True)

net = ResNet_Model(name='resnet34', num_classes=num_classes)
detector = None
loaded_checkpoint = torch.load(args.load, map_location='cpu')
net.load_state_dict(remove_data_parallel_prefix(extract_net_state(loaded_checkpoint)))
print('Model restored from {}'.format(args.load), flush=True)

detector_bundle_path = args.detector_load or infer_bundle_path(args.load)
if args.score == 'detector':
    if detector_bundle_path:
        detector_bundle = load_detector_bundle(detector_bundle_path)
    elif isinstance(loaded_checkpoint, dict) and 'detector' in loaded_checkpoint:
        detector_bundle = loaded_checkpoint
    else:
        raise FileNotFoundError('score=detector requires a detector bundle')
    detector_variant = detector_bundle.get('detector_variant', args.detector_variant)
    detector = build_detector(detector_variant)
    detector.load_state_dict(remove_data_parallel_prefix(detector_bundle['detector']))
    print('Detector restored from {}'.format(detector_bundle_path or args.load), flush=True)

net.eval()
device = torch.device('cuda' if args.ngpu > 0 and torch.cuda.is_available() else 'cpu')
net.to(device)
if detector is not None:
    detector.to(device)
    detector.eval()
cudnn.benchmark = True

ood_num_examples = args.ood_num_examples or (len(test_data) // 5)


def score_outputs(output):
    smax = F.softmax(output, dim=1)
    if args.score == 'energy':
        return -(args.T * torch.logsumexp(output / args.T, dim=1)).detach().cpu().numpy()
    if args.score == 'detector':
        detector_input = torch.logsumexp(output / args.T, dim=1).reshape(-1, 1)
        detector_output = detector(detector_input)
        return (detector_output[:, 0] - detector_output[:, 1]).detach().cpu().numpy()
    return -smax.max(dim=1).values.detach().cpu().numpy()


def get_id_scores(loader):
    all_scores = []
    right_scores = []
    wrong_scores = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            output = net(data)
            scores = score_outputs(output)
            preds = output.argmax(dim=1).detach().cpu().numpy()
            target_np = target.numpy()
            right_mask = preds == target_np
            wrong_mask = ~right_mask
            all_scores.append(scores)
            right_scores.append(scores[right_mask])
            wrong_scores.append(scores[wrong_mask])
    return (
        np.concatenate(all_scores).copy(),
        np.concatenate(right_scores).copy(),
        np.concatenate(wrong_scores).copy(),
    )


def build_ood_loader(name):
    if name == 'Texture':
        dataset = dset.ImageFolder(
            root=resolved_roots['textures_root'],
            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]),
        )
        workers = 4
        seed_offset = 11
    elif name == 'SVHN':
        dataset = svhn.SVHN(
            root=resolved_roots['svhn_root'],
            split='test',
            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]),
            download=args.download,
        )
        workers = 2
        seed_offset = 23
    elif name == 'Places365':
        dataset = dset.ImageFolder(
            root=resolved_roots['places_root'],
            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]),
        )
        workers = 2
        seed_offset = 37
    elif name == 'LSUN_C':
        dataset = dset.ImageFolder(
            root=resolved_roots['lsun_root'],
            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]),
        )
        workers = 1
        seed_offset = 41
    elif name == 'iSUN':
        dataset = dset.ImageFolder(
            root=resolved_roots['isun_root'],
            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]),
        )
        workers = 1
        seed_offset = 53
    else:
        raise ValueError(name)

    if len(dataset) > ood_num_examples:
        rng = np.random.default_rng(args.subset_seed + seed_offset)
        indices = rng.choice(len(dataset), size=ood_num_examples, replace=False)
        dataset = Subset(dataset, indices.tolist())
    return DataLoader(dataset, batch_size=args.test_bs, shuffle=False, num_workers=workers, pin_memory=True)


def get_ood_scores(loader):
    all_scores = []
    with torch.no_grad():
        for batch in loader:
            data = batch[0] if isinstance(batch, (tuple, list)) else batch
            data = data.to(device, non_blocking=True)
            output = net(data)
            all_scores.append(score_outputs(output))
    return np.concatenate(all_scores).copy()


def summarize_metric(point_triplet, boot_triplet):
    return {
        'fpr95': percentize(point_triplet[2], boot_triplet[2]),
        'auroc': percentize(point_triplet[0], boot_triplet[0]),
        'aupr': percentize(point_triplet[1], boot_triplet[1]),
    }


in_score, right_score, wrong_score = get_id_scores(test_loader)
error_point = get_measures(wrong_score, right_score)
error_boot = bootstrap_from_scores(wrong_score, right_score, args.bootstrap_trials, args.bootstrap_seed)

results = {
    'method_name': args.method_name,
    'load': args.load,
    'score': args.score,
    'bootstrap_trials': args.bootstrap_trials,
    'bootstrap_seed': args.bootstrap_seed,
    'subset_seed': args.subset_seed,
    'ood_num_examples': ood_num_examples,
    'error_rate': 100.0 * len(wrong_score) / (len(wrong_score) + len(right_score)),
    'error_detection': summarize_metric(error_point, error_boot),
    'datasets': {},
}

ood_names = ['Texture', 'SVHN', 'Places365', 'LSUN_C', 'iSUN']
ood_boot_arrays = {'fpr95': [], 'auroc': [], 'aupr': []}
ood_point_values = {'fpr95': [], 'auroc': [], 'aupr': []}

for idx, name in enumerate(ood_names):
    out_score = get_ood_scores(build_ood_loader(name))
    point = get_measures(-in_score, -out_score)
    boot = bootstrap_from_scores(-in_score, -out_score, args.bootstrap_trials, args.bootstrap_seed + (idx + 1) * 1000)
    dataset_result = summarize_metric(point, boot)
    results['datasets'][name] = dataset_result
    ood_point_values['fpr95'].append(dataset_result['fpr95']['point'])
    ood_point_values['auroc'].append(dataset_result['auroc']['point'])
    ood_point_values['aupr'].append(dataset_result['aupr']['point'])
    ood_boot_arrays['fpr95'].append(100.0 * boot[2])
    ood_boot_arrays['auroc'].append(100.0 * boot[0])
    ood_boot_arrays['aupr'].append(100.0 * boot[1])

overall = {}
for metric in ['fpr95', 'auroc', 'aupr']:
    boot_matrix = np.vstack(ood_boot_arrays[metric])
    overall[metric] = {
        'point': float(np.mean(ood_point_values[metric])),
        'mean': float(np.mean(np.mean(boot_matrix, axis=0))),
        'std': float(np.std(np.mean(boot_matrix, axis=0))),
    }
results['overall_average'] = overall

print('\nBootstrap evaluation for {}'.format(args.method_name), flush=True)
print('Error Rate: {:.2f}'.format(results['error_rate']), flush=True)
print('Error Detection: FPR95 {} | AUROC {} | AUPR {}'.format(
    format_triplet(results['error_detection']['fpr95']),
    format_triplet(results['error_detection']['auroc']),
    format_triplet(results['error_detection']['aupr']),
), flush=True)

for name in ood_names:
    entry = results['datasets'][name]
    print('{}: FPR95 {} | AUROC {} | AUPR {}'.format(
        name,
        format_triplet(entry['fpr95']),
        format_triplet(entry['auroc']),
        format_triplet(entry['aupr']),
    ), flush=True)

print('Overall Average: FPR95 {} | AUROC {} | AUPR {}'.format(
    format_triplet(results['overall_average']['fpr95']),
    format_triplet(results['overall_average']['auroc']),
    format_triplet(results['overall_average']['aupr']),
), flush=True)

if args.output_json:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as handle:
        json.dump(results, handle, indent=2)
    print('Saved JSON to {}'.format(args.output_json), flush=True)
