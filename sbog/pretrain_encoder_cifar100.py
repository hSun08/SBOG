# -*- coding: utf-8 -*-
import numpy as np
import os

import argparse
import time
import torch

import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

from resnet_anchor import ResNet_Model
from utils.validation_dataset import validation_split
from dataset_paths import default_cifar100_data_candidates, resolve_path


parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn', 'densenet'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--score', type=str, default='energy', help='OE|energy')
parser.add_argument('--add_slope', type=int, default=0)
parser.add_argument('--add_class', type=int, default=0)
parser.add_argument('--my_info', type=str, default='TODO')
parser.add_argument('--vanilla', type=int, default=0)
parser.add_argument('--oe', type=int, default=0)
parser.add_argument('--energy_weight', type=float, default=1)
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--download', action='store_true', help='Download CIFAR-100 if it is missing at --data_root.')
parser.add_argument('--token_embed_path', type=str, default='./token_embed_c100.npy')
parser.add_argument('--id_feat_output', type=str, default='./id_feat_cifar100_199epoch.npy')
parser.add_argument('--max_features_per_class', type=int, default=500)
args = parser.parse_args()


save_info = 'text_condition_c100'

args.save = args.save + save_info
if os.path.isdir(args.save) is False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)

args.data_root, searched_data_roots = resolve_path(args.data_root, default_cifar100_data_candidates())
print('Resolved CIFAR-100 data_root:', args.data_root)
if searched_data_roots:
    print('CIFAR-100 data root candidates:', searched_data_roots)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

train_data_in = dset.CIFAR100(args.data_root, train=True, transform=train_transform, download=args.download)
test_data = dset.CIFAR100(args.data_root, train=False, transform=test_transform, download=args.download)

if args.add_class:
    raise ValueError('--add_class is not supported in pretrain_cifar100.py because token embeddings cover 100 classes.')
num_classes = 100

calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

anchor = torch.from_numpy(np.load(args.token_embed_path)).cuda()
if anchor.shape[0] != num_classes:
    raise ValueError('Token embedding count {} does not match CIFAR-100 classes {}'.format(anchor.shape[0], num_classes))

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
net = ResNet_Model(name='resnet34', num_classes=num_classes)


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 0
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


# Restore model
model_found = False
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = args.load
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            model_found = True
            break
    if not model_found:
        assert False, 'could not find model to restore'

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    list(net.parameters()),
    state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,
        1e-6 / args.learning_rate))

criterion = torch.nn.CrossEntropyLoss()


def train_vanilla():
    net.train()
    loss_avg = 0.0

    for _, in_set in enumerate(train_loader_in):
        data = in_set[0]
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        x = net(data)
        x = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(x), 1, 1),
                                x.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1

        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    print(scheduler.get_lr())
    state['train_loss'] = loss_avg


if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

save_info = save_info + '_' + args.my_info

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                  '_' + save_info + '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train_vanilla()

    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                            '_' + save_info + '_epoch_' + str(epoch) + '.pt'))

    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                             '_' + save_info + '_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path):
        os.remove(prev_path)

    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                      '_' + save_info + '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss']
        ))

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'])
    )

number_dict = {}
for i in range(num_classes):
    number_dict[i] = 0
net.eval()
data_dict = torch.zeros(num_classes, args.max_features_per_class, 768).cuda()
with torch.no_grad():
    for _, in_set in enumerate(train_loader_in):
        data = in_set[0]
        target = in_set[1]
        data, target = data.cuda(), target.cuda()
        feat = net(data)
        target_numpy = target.cpu().data.numpy()
        for index in range(len(target)):
            dict_key = target_numpy[index]
            if number_dict[dict_key] < args.max_features_per_class:
                data_dict[dict_key][number_dict[dict_key]] = feat[index].detach()
                number_dict[dict_key] += 1

np.save(args.id_feat_output, data_dict.cpu().numpy())
print('Saved CIFAR-100 ID features to {}'.format(args.id_feat_output))
