#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn

from dataset_paths import default_cifar100_data_candidates, resolve_path
from resnet_anchor import ResNet_Model
from robust_alignment_utils import (
    MultiViewCIFAR100,
    compute_class_logits,
    compute_robust_logits,
    get_current_robust_weight,
)
from utils.validation_dataset import validation_split


def build_parser():
    parser = argparse.ArgumentParser(
        description='CIFAR-100 step1 robust alignment for outlier-energy.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model', '-m', type=str, default='wrn')
    parser.add_argument('--calibration', '-c', action='store_true')
    parser.add_argument('--epochs', '-e', type=int, default=80)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.08)
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', '-d', type=float, default=0.0005)
    parser.add_argument('--save', '-s', type=str, default='./snapshots/')
    parser.add_argument('--load', '-l', type=str, default='')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--my_info', type=str, default='ra2rw010')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--token_embed_path', type=str, default='./token_embed_c100.npy')
    parser.add_argument('--id_feat_output', type=str, default='./id_feat_cifar100_ra2rw010.npy')
    parser.add_argument('--max_features_per_class', type=int, default=500)
    parser.add_argument('--logit_temperature', type=float, default=0.1)
    parser.add_argument('--num_views', type=int, default=2)
    parser.add_argument('--robust_weight', type=float, default=0.10)
    parser.add_argument('--robust_start_epoch', type=int, default=10)
    parser.add_argument('--robust_ramp_epochs', type=int, default=15)
    parser.add_argument('--robust_temperature', type=float, default=1.0)
    parser.add_argument('--robust_radius', type=float, default=0.03)
    parser.add_argument('--max_train_steps', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--skip_id_feat_dump', action='store_true')
    return parser


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def main():
    parser = build_parser()
    args = parser.parse_args()

    save_info = 'text_condition_c100_robust'
    save_dir = os.path.join(args.save, save_info)
    os.makedirs(save_dir, exist_ok=True)

    state = {k: v for k, v in args._get_kwargs()}
    state['save'] = save_dir
    print(state, flush=True)

    torch.manual_seed(1)
    np.random.seed(args.seed)

    args.data_root, searched_data_roots = resolve_path(args.data_root, default_cifar100_data_candidates())
    print('Resolved CIFAR-100 data_root:', args.data_root, flush=True)
    if searched_data_roots:
        print('CIFAR-100 data root candidates:', searched_data_roots, flush=True)

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = trn.Compose([
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(32, padding=4),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])

    train_data = MultiViewCIFAR100(
        args.data_root,
        train=True,
        transform=train_transform,
        num_views=args.num_views,
        download=args.download,
    )
    feature_data = dset.CIFAR100(
        args.data_root,
        train=True,
        transform=train_transform,
        download=args.download,
    )

    calib_indicator = ''
    if args.calibration:
        train_data, _ = validation_split(train_data, val_share=0.1)
        calib_indicator = '_calib'

    num_classes = 100
    anchor = torch.from_numpy(np.load(args.token_embed_path)).float().cuda()
    if anchor.shape[0] != num_classes:
        raise ValueError(
            'Token embedding count {} does not match CIFAR-100 classes {}'.format(anchor.shape[0], num_classes)
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True,
    )
    feature_loader = torch.utils.data.DataLoader(
        feature_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True,
    )

    net = ResNet_Model(name='resnet34', num_classes=num_classes)
    if args.load:
        checkpoint = torch.load(args.load, map_location='cpu')
        net.load_state_dict(checkpoint, strict=True)
        print('Loaded init checkpoint from {}'.format(args.load), flush=True)
    else:
        print('No init checkpoint specified. Training will start from scratch.', flush=True)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True

    optimizer = torch.optim.SGD(
        net.parameters(),
        state['learning_rate'],
        momentum=state['momentum'],
        weight_decay=state['decay'],
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * max(len(train_loader), 1),
            1,
            1e-6 / args.learning_rate,
        ),
    )
    criterion = torch.nn.CrossEntropyLoss()

    save_info = save_info + '_' + args.my_info
    training_csv = os.path.join(
        save_dir,
        args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_info + '_training_results.csv',
    )
    with open(training_csv, 'w') as handle:
        handle.write('epoch,time(s),train_loss,nominal_loss,robust_loss,robust_weight\n')

    print('Beginning Training\n', flush=True)
    for epoch in range(args.epochs):
        net.train()
        current_robust_weight = get_current_robust_weight(
            epoch=epoch,
            base_weight=args.robust_weight,
            start_epoch=args.robust_start_epoch,
            ramp_epochs=args.robust_ramp_epochs,
        )
        loss_avg = 0.0
        nominal_avg = 0.0
        robust_avg = 0.0
        begin_epoch = time.time()

        for step, (view_batch, target) in enumerate(train_loader, start=1):
            view_batch = view_batch.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            batch_size, num_views, channels, height, width = view_batch.shape

            feat = net(view_batch.view(batch_size * num_views, channels, height, width))
            feat = feat.view(batch_size, num_views, -1)

            nominal_logits = compute_class_logits(feat[:, 0], anchor, args.logit_temperature)
            nominal_loss = criterion(nominal_logits, target)

            robust_loss = nominal_loss.new_zeros(())
            if current_robust_weight > 0:
                view_logits = compute_class_logits(feat, anchor, args.logit_temperature)
                robust_logits = compute_robust_logits(
                    view_logits,
                    robust_temperature=args.robust_temperature,
                    robust_radius=args.robust_radius,
                )
                robust_loss = criterion(robust_logits, target)
                loss = (1.0 - current_robust_weight) * nominal_loss + current_robust_weight * robust_loss
            else:
                loss = nominal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_avg = loss_avg * 0.8 + float(loss) * 0.2
            nominal_avg = nominal_avg * 0.8 + float(nominal_loss) * 0.2
            robust_avg = robust_avg * 0.8 + float(robust_loss) * 0.2

            if step == 1 or step % args.log_interval == 0 or step == len(train_loader):
                print(
                    'Epoch {} Step {}/{} | loss {:.4f} | nominal {:.4f} | robust {:.4f} | rw {:.3f}'.format(
                        epoch + 1,
                        step,
                        len(train_loader),
                        float(loss),
                        float(nominal_loss),
                        float(robust_loss),
                        current_robust_weight,
                    ),
                    flush=True,
                )
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break

        ckpt_path = os.path.join(
            save_dir,
            args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_info + '_epoch_' + str(epoch) + '.pt',
        )
        torch.save(net.state_dict(), ckpt_path)
        if epoch % 10 != 0:
            prev_path = os.path.join(
                save_dir,
                args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_info + '_epoch_' + str(epoch - 1) + '.pt',
            )
            if os.path.exists(prev_path):
                os.remove(prev_path)

        with open(training_csv, 'a') as handle:
            handle.write('%03d,%05d,%0.6f,%0.6f,%0.6f,%0.4f\n' % (
                epoch + 1,
                time.time() - begin_epoch,
                loss_avg,
                nominal_avg,
                robust_avg,
                current_robust_weight,
            ))

        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Nominal {3:.4f} | Robust {4:.4f} | RW {5:.3f}'.format(
                epoch + 1,
                int(time.time() - begin_epoch),
                loss_avg,
                nominal_avg,
                robust_avg,
                current_robust_weight,
            ),
            flush=True,
        )

    if args.skip_id_feat_dump:
        print('Skipped ID feature dump', flush=True)
        return

    number_dict = {i: 0 for i in range(num_classes)}
    net.eval()
    data_dict = torch.zeros(num_classes, args.max_features_per_class, 768).cuda()
    with torch.no_grad():
        for data, target in feature_loader:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            feat = net(data)
            target_numpy = target.cpu().numpy()
            for index in range(len(target)):
                class_index = int(target_numpy[index])
                if number_dict[class_index] < args.max_features_per_class:
                    data_dict[class_index][number_dict[class_index]] = feat[index].detach()
                    number_dict[class_index] += 1

    np.save(args.id_feat_output, data_dict.cpu().numpy())
    print('Saved ID features to {}'.format(args.id_feat_output), flush=True)


if __name__ == '__main__':
    main()
