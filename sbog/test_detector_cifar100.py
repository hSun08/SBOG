import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn

from resnet import ResNet_Model
from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
import utils.svhn_loader as svhn
import utils.score_calculation as lib
from dataset_paths import (
    default_cifar100_data_candidates,
    default_cifar100_ood_candidates,
    format_missing_path_message,
    remove_data_parallel_prefix,
    resolve_path,
)

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--detector_load', type=str, default='', help='Optional detector bundle checkpoint for score=detector.')
parser.add_argument('--detector_variant', choices=['default', 'imagenet'], default='imagenet',
                    help='Detector architecture to use when loading a standalone detector state dict.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='energy', type=str, help='score options: MSP|energy|detector')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--download', action='store_true', help='Download CIFAR-100/SVHN if they are missing.')
parser.add_argument('--textures_root', type=str, default='')
parser.add_argument('--svhn_root', type=str, default='')
parser.add_argument('--places_root', type=str, default='')
parser.add_argument('--lsun_root', type=str, default='')
parser.add_argument('--isun_root', type=str, default='')
args = parser.parse_args()
print(args)

args.data_root, searched_data_roots = resolve_path(args.data_root, default_cifar100_data_candidates())
print('Resolved CIFAR-100 data_root:', args.data_root)
if searched_data_roots:
    print('CIFAR-100 data root candidates:', searched_data_roots)

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
    print('Resolved {}: {}'.format(flag, resolved))
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

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)


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


net = ResNet_Model(name='resnet34', num_classes=num_classes)
detector = None
start_epoch = 0

if args.load != '':
    if not os.path.isfile(args.load):
        raise FileNotFoundError('Missing checkpoint: {}'.format(args.load))
    loaded_checkpoint = torch.load(args.load, map_location='cpu')
    net.load_state_dict(remove_data_parallel_prefix(extract_net_state(loaded_checkpoint)))
    print('Model restored from {}'.format(args.load))
    start_epoch = 1
else:
    loaded_checkpoint = None

detector_bundle_path = args.detector_load or infer_bundle_path(args.load)
if args.score == 'detector':
    if detector_bundle_path:
        detector_bundle = load_detector_bundle(detector_bundle_path)
    elif isinstance(loaded_checkpoint, dict) and 'detector' in loaded_checkpoint:
        detector_bundle = loaded_checkpoint
    else:
        raise FileNotFoundError(
            'score=detector requires a detector bundle. Pass --detector_load or load a *_bundle.pt checkpoint.'
        )
    detector_variant = detector_bundle.get('detector_variant', args.detector_variant)
    detector = build_detector(detector_variant)
    detector.load_state_dict(remove_data_parallel_prefix(detector_bundle['detector']))
    print('Detector restored from {}'.format(detector_bundle_path or args.load))

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    if detector is not None:
        detector.cuda()

if detector is not None:
    detector.eval()

cudnn.benchmark = True

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                batch_scores = to_np((output.mean(1) - torch.logsumexp(output, dim=1)))
            elif args.score == 'energy':
                batch_scores = -to_np((args.T * torch.logsumexp(output / args.T, dim=1)))
            elif args.score == 'detector':
                detector_input = torch.logsumexp(output / args.T, dim=1).reshape(-1, 1)
                detector_output = detector(detector_input)
                batch_scores = to_np(detector_output[:, 0] - detector_output[:, 1])
            else:
                batch_scores = -np.max(smax, axis=1)
            _score.append(batch_scores)

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                if args.use_xent or args.score in {'energy', 'detector'}:
                    _right_score.append(batch_scores[right_indices])
                    _wrong_score.append(batch_scores[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    return concat(_score)[:ood_num_examples].copy()


if args.score == 'Odin':
    in_score, right_score, wrong_score = lib.get_ood_scores_odin(test_loader, net, args.test_bs, ood_num_examples,
                                                                 args.T, args.noise, in_dist=True)
elif args.score == 'M':
    from torch.autograd import Variable

    _, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    train_data = dset.CIFAR100(args.data_root, train=True, transform=test_transform, download=args.download)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
                                               num_workers=args.prefetch, pin_memory=True)
    num_batches = ood_num_examples // args.test_bs

    temp_x = torch.rand(2, 3, 32, 32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = net.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader)
    in_score = lib.get_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count - 1, args.noise,
                                         num_batches, in_dist=True)
    print(in_score[-3:], in_score[-103:-100])
else:
    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.method_name)

auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):
    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, ood_num_examples, args.T, args.noise)
        elif args.score == 'M':
            out_score = lib.get_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count - 1,
                                                  args.noise, num_batches)
        else:
            out_score = get_ood_scores(ood_loader)

        if args.out_as_pos:
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0])
        auprs.append(measures[1])
        fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)


ood_data = dset.ImageFolder(root=resolved_roots['textures_root'],
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\nTexture Detection')
get_and_print_results(ood_loader)

ood_data = svhn.SVHN(root=resolved_roots['svhn_root'], split='test',
                     transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]), download=args.download)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nSVHN Detection')
get_and_print_results(ood_loader)

ood_data = dset.ImageFolder(root=resolved_roots['places_root'],
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nPlaces365 Detection')
get_and_print_results(ood_loader)

ood_data = dset.ImageFolder(root=resolved_roots['lsun_root'],
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nLSUN_C Detection')
get_and_print_results(ood_loader)

ood_data = dset.ImageFolder(root=resolved_roots['isun_root'],
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\niSUN Detection')
get_and_print_results(ood_loader)
