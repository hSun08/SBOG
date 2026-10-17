# -*- coding: utf-8 -*-
import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as trn

from resnet_anchor import ResNet_Model
from robust_alignment_utils import (
    MultiViewImageFolder,
    compute_class_logits,
    compute_robust_logits,
    get_current_robust_weight,
)
from utils.validation_dataset import validation_split


parser = argparse.ArgumentParser(
    description='ImageNet-100 encoder alignment with the Sinkhorn-DRO robust auxiliary loss (SBOG step 1, robust branch)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--dataset', type=str, default='in100')
parser.add_argument('--model', '-m', type=str, default='wrn')
parser.add_argument('--calibration', '-c', action='store_true')
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
parser.add_argument('--batch_size', '-b', type=int, default=160)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', '-d', type=float, default=0.0005)
parser.add_argument('--save', '-s', type=str, default='./snapshots/')
parser.add_argument('--load', '-l', type=str, default='')
parser.add_argument('--test', '-t', action='store_true')
parser.add_argument('--ngpu', type=int, default=8)
parser.add_argument('--prefetch', type=int, default=4)
parser.add_argument('--my_info', type=str, default='TODO')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--token_embed_path', type=str, default='./token_embed_in100.npy')
parser.add_argument('--id_feat_output', type=str, default='./id_feat_in100_99epoch.npy')
parser.add_argument('--max_features_per_class', type=int, default=1000)
parser.add_argument('--logit_temperature', type=float, default=0.1)
parser.add_argument('--num_views', type=int, default=3)
parser.add_argument('--robust_weight', type=float, default=0.25)
parser.add_argument('--robust_start_epoch', type=int, default=10)
parser.add_argument('--robust_ramp_epochs', type=int, default=5)
parser.add_argument('--robust_temperature', type=float, default=1.0)
parser.add_argument('--robust_radius', type=float, default=0.05)
parser.add_argument('--max_train_steps', type=int, default=0)
parser.add_argument('--skip_id_feat_dump', action='store_true')
args = parser.parse_args()

save_info = 'text_condition_in100_robust'
args.save = args.save + save_info
if os.path.isdir(args.save) is False:
    os.makedirs(args.save, exist_ok=True)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)

traindir = os.path.join(args.data_root, 'train')
valdir = os.path.join(args.data_root, 'val')
normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
train_transform = trn.Compose([
    trn.RandomResizedCrop(224),
    trn.RandomHorizontalFlip(),
    trn.ToTensor(),
    normalize,
])
test_transform = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224),
    trn.ToTensor(),
    normalize,
])

train_data_in = MultiViewImageFolder(
    traindir,
    transform=train_transform,
    num_views=args.num_views,
)
feature_data_in = torchvision.datasets.ImageFolder(
    traindir,
    train_transform,
)
test_data = torchvision.datasets.ImageFolder(
    valdir,
    test_transform,
)
num_classes = len(feature_data_in.classes)

calib_indicator = ''
if args.calibration:
    train_data_in, _ = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

anchor = torch.from_numpy(np.load(args.token_embed_path)).float().cuda()
if anchor.shape[0] != num_classes:
    raise ValueError(
        'Token embedding count {} does not match dataset classes {}'.format(anchor.shape[0], num_classes)
    )

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.prefetch,
    pin_memory=True,
)
feature_loader_in = torch.utils.data.DataLoader(
    feature_data_in,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.prefetch,
    pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.prefetch,
    pin_memory=True,
)

net = ResNet_Model(name='resnet34', num_classes=num_classes)


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


model_found = False
if args.load != '':
    model_name = args.load
    if os.path.isfile(model_name):
        net.load_state_dict(remove_data_parallel(torch.load(model_name)))
        print('Model restored!')
        model_found = True
    if not model_found:
        raise AssertionError('could not find model to restore')

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True

optimizer = torch.optim.SGD(
    list(net.parameters()),
    state['learning_rate'],
    momentum=state['momentum'],
    weight_decay=state['decay'],
    nesterov=True,
)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,
        1e-6 / args.learning_rate,
    ),
)


criterion = torch.nn.CrossEntropyLoss()


def train_robust(epoch):
    net.train()
    loss_avg = 0.0
    nominal_avg = 0.0
    robust_avg = 0.0
    current_robust_weight = get_current_robust_weight(
        epoch=epoch,
        base_weight=args.robust_weight,
        start_epoch=args.robust_start_epoch,
        ramp_epochs=args.robust_ramp_epochs,
    )

    for step, (view_batch, target) in enumerate(train_loader_in, start=1):
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

        if args.max_train_steps > 0 and step >= args.max_train_steps:
            break

    print(scheduler.get_last_lr())
    print('robust_weight_current:', current_robust_weight)
    state['train_loss'] = loss_avg
    state['train_nominal_loss'] = nominal_avg
    state['train_robust_loss'] = robust_avg
    state['robust_weight_current'] = current_robust_weight


if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

save_info = save_info + '_' + args.my_info
training_csv = os.path.join(
    args.save,
    args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_info + '_training_results.csv',
)
with open(training_csv, 'w') as f:
    f.write('epoch,time(s),train_loss,nominal_loss,robust_loss,robust_weight\n')

print('Beginning Training\n')
for epoch in range(0, args.epochs):
    state['epoch'] = epoch
    begin_epoch = time.time()
    train_robust(epoch)

    ckpt_path = os.path.join(
        args.save,
        args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_info + '_epoch_' + str(epoch) + '.pt',
    )
    torch.save(net.state_dict(), ckpt_path)

    if epoch % 10 != 0:
        prev_path = os.path.join(
            args.save,
            args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + '_' + save_info + '_epoch_' + str(epoch - 1) + '.pt',
        )
        if os.path.exists(prev_path):
            os.remove(prev_path)

    with open(training_csv, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.6f,%0.6f,%0.4f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['train_nominal_loss'],
            state['train_robust_loss'],
            state['robust_weight_current'],
        ))

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Nominal {3:.4f} | Robust {4:.4f} | RW {5:.3f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['train_nominal_loss'],
        state['train_robust_loss'],
        state['robust_weight_current'],
    ))

if not args.skip_id_feat_dump:
    number_dict = {i: 0 for i in range(num_classes)}
    net.eval()
    data_dict = torch.zeros(num_classes, args.max_features_per_class, 768).cuda()
    with torch.no_grad():
        for data, target in feature_loader_in:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            feat = net(data)
            target_numpy = target.cpu().numpy()
            for index in range(len(target)):
                dict_key = int(target_numpy[index])
                if number_dict[dict_key] < args.max_features_per_class:
                    data_dict[dict_key][number_dict[dict_key]] = feat[index].detach()
                    number_dict[dict_key] += 1

    np.save(args.id_feat_output, data_dict.cpu().numpy())
    print('Saved ID features to {}'.format(args.id_feat_output))
else:
    print('Skipped ID feature dump')
