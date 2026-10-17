# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn

from dataset_paths import (
    default_cifar100_data_candidates,
    format_missing_path_message,
    remove_data_parallel_prefix,
    resolve_generated_ood_root,
    resolve_path,
)
from resnet import ResNet_Model
from utils.validation_dataset import validation_split


def build_parser():
    parser = argparse.ArgumentParser(
        description='Tunes a CIFAR Classifier with OE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument('--model', '-m', type=str, default='wrn',
                        choices=['allconv', 'wrn', 'densenet'], help='Choose architecture.')
    parser.add_argument('--detector_variant', choices=['default', 'imagenet'], default='default',
                        help='default keeps the original CIFAR detector; imagenet uses the stronger ImageNet-style detector/training recipe.')
    parser.add_argument('--calibration', '-c', action='store_true',
                        help='Train a model to be used for calibration. This holds out some data for validation.')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
    parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, default='',
                        help='Checkpoint path or directory to resume / initialize from.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--m_in', type=float, default=-25.,
                        help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7.,
                        help='margin for out-distribution; below this value will be penalized')
    parser.add_argument('--score', type=str, default='energy', help='OE|energy')
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--add_slope', type=int, default=0)
    parser.add_argument('--add_class', type=int, default=0)
    parser.add_argument('--my_info', type=str, default='')
    parser.add_argument('--vanilla', type=int, default=0)
    parser.add_argument('--godin', type=int, default=0)
    parser.add_argument('--gan', type=int, default=0)
    parser.add_argument('--oe', type=int, default=0)
    parser.add_argument('--r50', type=int, default=0)
    parser.add_argument('--vos', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--additional_info', type=str, default='')
    parser.add_argument('--energy_weight', type=float, default=2.5)
    parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--download', action='store_true', help='Download CIFAR-100 if it is missing at --data_root.')
    parser.add_argument('--generated_ood_root', type=str, default='')
    parser.add_argument('--ood_root', type=str, default='')
    parser.add_argument('--start_epoch', type=int, default=40)
    parser.add_argument('--sample_number', type=int, default=1000)
    parser.add_argument('--select', type=int, default=1)
    parser.add_argument('--sample_from', type=int, default=10000)
    parser.add_argument('--loss_weight', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Gradient clipping norm. Set <=0 to disable.')
    parser.add_argument('--max_steps_per_epoch', type=int, default=0,
                        help='Debug option to limit steps per epoch. 0 disables the limit.')
    parser.add_argument('--save_detector_bundle', type=int, default=1,
                        help='Also save a companion bundle checkpoint containing the detector head.')
    return parser


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def load_initial_weights(model, checkpoint_path):
    pretrained_weights = torch.load(checkpoint_path, map_location='cpu')
    pretrained_weights = remove_data_parallel_prefix(pretrained_weights)
    try:
        model.load_state_dict(pretrained_weights, strict=True)
        print('Loaded init checkpoint from {} (strict)'.format(checkpoint_path), flush=True)
        return
    except RuntimeError:
        model_state = model.state_dict()
        matched = {
            key: value for key, value in pretrained_weights.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        if not matched:
            raise RuntimeError(
                'Failed to initialize from {} because no parameters matched by name and shape'.format(
                    checkpoint_path
                )
            )
        model_state.update(matched)
        model.load_state_dict(model_state, strict=False)
        print(
            'Loaded init checkpoint from {} (partial: {} / {} tensors)'.format(
                checkpoint_path, len(matched), len(model_state)
            ),
            flush=True,
        )


def maybe_load_checkpoint(model, args, calib_indicator):
    if not args.load:
        print('No init checkpoint specified. Training will start from scratch.', flush=True)
        return
    if os.path.isfile(args.load):
        load_initial_weights(model, args.load)
        return

    model_found = False
    for i in range(999, -1, -1):
        model_name = os.path.join(
            args.load,
            args.dataset + calib_indicator + '_' + args.model + '_pretrained_epoch_' + str(i) + '.pt',
        )
        if args.add_class:
            model_name = os.path.join(
                './snapshots/baseline',
                args.dataset + calib_indicator + '_' + args.model + '_baseline_epoch_' + str(i) + '.pt',
            )
        if os.path.isfile(model_name):
            load_initial_weights(model, model_name)
            print('Model restored! Epoch:', i, flush=True)
            model_found = True
            break
    if not model_found:
        raise FileNotFoundError('could not find model to restore from {}'.format(args.load))


def unwrap_module(module):
    if isinstance(module, torch.nn.DataParallel):
        return module.module
    return module


def bundle_path_from_checkpoint(ckpt_path):
    root, ext = os.path.splitext(ckpt_path)
    return root + '_bundle' + ext


def save_epoch_checkpoint(net, detector, ckpt_path, args, epoch):
    torch.save(net.state_dict(), ckpt_path)
    if not args.save_detector_bundle:
        return
    bundle_payload = {
        'net': unwrap_module(net).state_dict(),
        'detector': unwrap_module(detector).state_dict(),
        'detector_variant': args.detector_variant,
        'score': args.score,
        'T': args.T,
        'epoch': epoch,
        'args': {k: v for k, v in vars(args).items()},
    }
    torch.save(bundle_payload, bundle_path_from_checkpoint(ckpt_path))


def build_dataloaders(args):
    args.data_root, searched_data_roots = resolve_path(args.data_root, default_cifar100_data_candidates())
    print('Resolved CIFAR-100 data_root:', args.data_root, flush=True)
    if searched_data_roots:
        print('CIFAR-100 data root candidates:', searched_data_roots, flush=True)

    ood_root, searched_ood_roots = resolve_generated_ood_root(args.ood_root, args.generated_ood_root, args.my_info)
    print('Resolved generated OOD root:', ood_root, flush=True)
    if searched_ood_roots:
        print('Generated OOD root candidates:', searched_ood_roots, flush=True)
    if not ood_root or not os.path.isdir(ood_root):
        raise FileNotFoundError(format_missing_path_message('generated CIFAR-100 OOD root', searched_ood_roots))

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = trn.Compose([
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(32, padding=4),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    ood_transform = trn.Compose([
        trn.ToTensor(),
        trn.ToPILImage(),
        trn.RandomCrop(32, padding=4),
        trn.RandomHorizontalFlip(),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])

    train_data_in = dset.CIFAR100(args.data_root, train=True, transform=train_transform, download=args.download)
    test_data = dset.CIFAR100(args.data_root, train=False, transform=test_transform, download=args.download)

    calib_indicator = ''
    if args.calibration:
        train_data_in, _ = validation_split(train_data_in, val_share=0.1)
        calib_indicator = '_calib'

    ood_data = dset.ImageFolder(root=ood_root, transform=ood_transform)
    print('Train ID samples:', len(train_data_in), flush=True)
    print('Train OOD samples:', len(ood_data), flush=True)
    print('Test ID samples:', len(test_data), flush=True)

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True,
    )
    train_loader_out = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.oe_batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=True,
    )
    return train_loader_in, train_loader_out, test_loader, calib_indicator


def build_detector(args):
    if args.detector_variant == 'imagenet':
        return torch.nn.Sequential(
            torch.nn.Linear(1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
        ).cuda()
    return torch.nn.Sequential(
        torch.nn.Linear(1, 2)
    ).cuda()


def train_epoch_default(args, state, net, detector, optimizer, scheduler, criterion, train_loader_in, train_loader_out):
    net.train()
    detector.train()
    loss_avg = 0.0
    loss_energy_avg = 0.0
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))

    for step, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out), start=1):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        x = net(data)

        optimizer.zero_grad()
        energy_term = torch.tensor(0.0, device=x.device)

        if args.add_class:
            target = torch.cat([target, torch.ones(len(out_set[0]), device=target.device).long() * (100 if args.add_class else 99)], -1)
            loss = F.cross_entropy(x, target)
        else:
            loss = F.cross_entropy(x[:len(in_set[0])], target)
            if args.score == 'energy':
                Ec_out = torch.logsumexp(x[len(in_set[0]):], dim=1)
                Ec_in = torch.logsumexp(x[:len(in_set[0])], dim=1)
                binary_labels = torch.ones(len(x), device=x.device)
                binary_labels[len(in_set[0]):] = 0

                input_for_lr = torch.cat((Ec_in, Ec_out), -1)
                output1 = detector(input_for_lr.reshape(-1, 1))
                energy_reg_loss = criterion(output1, binary_labels.long())
                energy_term = args.energy_weight * energy_reg_loss
                loss += energy_term
            elif args.score == 'OE':
                oe_term = -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
                energy_term = args.energy_weight * oe_term
                loss += energy_term

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        loss_energy_avg = loss_energy_avg * 0.8 + float(energy_term) * 0.2
        if args.max_steps_per_epoch > 0 and step >= args.max_steps_per_epoch:
            print('Stopping epoch {} early at step {} due to --max_steps_per_epoch'.format(state.get('epoch', 0) + 1, step), flush=True)
            break

    print(scheduler.get_last_lr(), flush=True)
    print('loss energy is: ', loss_energy_avg, flush=True)
    state['train_loss'] = loss_avg
    state['train_energy_loss'] = loss_energy_avg


def train_epoch_imagenet(args, state, net, detector, optimizer, scheduler, criterion, train_loader_in, train_loader_out):
    net.train()
    detector.train()
    loss_avg = 0.0
    loss_energy_avg = 0.0
    epoch_num = state.get('epoch', 0) + 1
    total_steps = len(train_loader_in)

    batch_iterator = iter(train_loader_out)
    for step, in_set in enumerate(train_loader_in, start=1):
        try:
            out_set = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_loader_out)
            out_set = next(batch_iterator)

        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        permutation_idx = torch.randperm(len(data), device=data.device)
        fake_target = torch.cat(
            [target, torch.full((len(out_set[0]),), -1, device=target.device, dtype=torch.long)],
            dim=0,
        )
        binary_labels = torch.ones(len(data), device=data.device, dtype=torch.long)
        binary_labels[len(in_set[0]):] = 0

        x = net(data[permutation_idx])
        keep_mask = binary_labels[permutation_idx].bool()
        Ec_out = torch.logsumexp(x[~keep_mask], dim=1) / args.T
        Ec_in = torch.logsumexp(x[keep_mask], dim=1) / args.T

        optimizer.zero_grad()
        ce_loss = F.cross_entropy(x[keep_mask], fake_target[permutation_idx][keep_mask])
        detector_input = torch.cat((Ec_in, Ec_out), dim=0).reshape(-1, 1)
        detector_output = detector(detector_input)
        energy_reg_loss = criterion(detector_output, binary_labels)
        loss = ce_loss + args.energy_weight * energy_reg_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                'Non-finite loss at epoch {} step {}: loss={} ce={} energy={} logit_abs_max={}'.format(
                    epoch_num,
                    step,
                    float(loss.detach().cpu()),
                    float(ce_loss.detach().cpu()),
                    float((args.energy_weight * energy_reg_loss).detach().cpu()),
                    float(x.detach().abs().max().cpu()),
                )
            )

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(net.parameters()) + list(detector.parameters()), args.grad_clip
            )
        optimizer.step()
        scheduler.step()

        loss_energy_avg = loss_energy_avg * 0.8 + float(args.energy_weight * energy_reg_loss) * 0.2
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        if step == 1 or step % args.log_interval == 0 or step == total_steps:
            print(
                'Epoch {} Step {}/{} | loss {:.4f} | ce {:.4f} | energy {:.4f}'.format(
                    epoch_num, step, total_steps, float(loss), float(ce_loss), float(args.energy_weight * energy_reg_loss)
                ),
                flush=True,
            )
        if args.max_steps_per_epoch > 0 and step >= args.max_steps_per_epoch:
            print('Stopping epoch {} early at step {} due to --max_steps_per_epoch'.format(epoch_num, step), flush=True)
            break

    print(scheduler.get_last_lr(), flush=True)
    print('loss energy is: ', loss_energy_avg, flush=True)
    state['train_loss'] = loss_avg
    state['train_energy_loss'] = loss_energy_avg


def evaluate(state, net, test_loader):
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = net(data)
            loss = F.cross_entropy(output, target)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.score not in {'OE', 'energy'}:
        raise ValueError('Unsupported score {}'.format(args.score))

    save_info = 'energy_ft_sd'
    args.save = args.save + save_info
    if not os.path.isdir(args.save):
        os.makedirs(args.save, exist_ok=True)

    state = {k: v for k, v in args._get_kwargs()}
    print(state, flush=True)

    torch.manual_seed(1)
    np.random.seed(args.seed)

    train_loader_in, train_loader_out, test_loader, calib_indicator = build_dataloaders(args)
    num_classes = 101 if args.add_class else 100

    if args.r50:
        net = ResNet_Model(name='resnet50', num_classes=num_classes)
    else:
        net = ResNet_Model(name='resnet34', num_classes=num_classes)

    maybe_load_checkpoint(net, args, calib_indicator)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True

    detector = build_detector(args)
    if args.ngpu > 1:
        detector = torch.nn.DataParallel(detector, device_ids=list(range(args.ngpu)))

    optimizer = torch.optim.SGD(
        list(net.parameters()) + list(detector.parameters()),
        state['learning_rate'],
        momentum=state['momentum'],
        weight_decay=state['decay'],
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * max(len(train_loader_in), 1),
            1,
            1e-6 / args.learning_rate,
        ),
    )
    criterion = torch.nn.CrossEntropyLoss()

    if args.test:
        evaluate(state, net, test_loader)
        print(state, flush=True)
        return

    save_tag = save_info + '_slope_' + str(args.add_slope) + '_' + 'weight_' + str(args.energy_weight)
    save_tag = save_tag + '_' + args.my_info + '_' + args.additional_info
    results_path = os.path.join(
        args.save,
        args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_tag + '_training_results.csv',
    )
    with open(results_path, 'w') as handle:
        handle.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    print('Beginning Training\n', flush=True)
    for epoch in range(args.epochs):
        state['epoch'] = epoch
        begin_epoch = time.time()

        if args.detector_variant == 'imagenet':
            train_epoch_imagenet(args, state, net, detector, optimizer, scheduler, criterion, train_loader_in, train_loader_out)
        else:
            train_epoch_default(args, state, net, detector, optimizer, scheduler, criterion, train_loader_in, train_loader_out)
        evaluate(state, net, test_loader)

        ckpt_path = os.path.join(
            args.save,
            args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_tag + '_epoch_' + str(epoch) + '.pt',
        )
        save_epoch_checkpoint(net, detector, ckpt_path, args, epoch)

        prev_path = os.path.join(
            args.save,
            args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_tag + '_epoch_' + str(epoch - 1) + '.pt',
        )
        if os.path.exists(prev_path):
            os.remove(prev_path)
        prev_bundle_path = bundle_path_from_checkpoint(prev_path)
        if os.path.exists(prev_bundle_path):
            os.remove(prev_bundle_path)

        with open(results_path, 'a') as handle:
            handle.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['test_loss'],
                100 - 100. * state['test_accuracy'],
            ))

        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                state['train_loss'],
                state['test_loss'],
                100 - 100. * state['test_accuracy'],
            ),
            flush=True,
        )


if __name__ == '__main__':
    main()
