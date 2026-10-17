#!/usr/bin/env python3
"""Plain cross-entropy pre-training of the ResNet-34 ID classifier on ImageNet-100.

The checkpoint it writes is the initialisation (`--load`) of `train_detector_in100.py` (SBOG step 4), i.e. the
"CE-pretrained model" used by the outlier-exposure fine-tuning. Every epoch is kept so a specific epoch can be chosen.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as trn

from resnet import ResNet_Model


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def main():
    p = argparse.ArgumentParser(description='Vanilla CE pre-training of ResNet-34 on ImageNet-100 (init for step 4)')
    p.add_argument('--data_root', type=str, required=True, help='ImageFolder root with train/ and val/')
    p.add_argument('--save', type=str, required=True)
    p.add_argument('--my_info', type=str, default='custom100')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)   # this ResNet-34 keeps 224px feature maps; bs 128 OOMs on 40GB
    p.add_argument('--test_bs', type=int, default=200)
    p.add_argument('--learning_rate', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--decay', type=float, default=0.0005)
    p.add_argument('--prefetch', type=int, default=8)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--max_steps_per_epoch', type=int, default=0, help='>0 truncates every epoch (smoke tests only)')
    args = p.parse_args()
    print(vars(args), flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'train'), trn.Compose([
        trn.RandomResizedCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor(), normalize]))
    test_data = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'val'), trn.Compose([
        trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), normalize]))
    num_classes = len(train_data.classes)
    print('classes', num_classes, 'train', len(train_data), 'val', len(test_data), flush=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    net = ResNet_Model(name='resnet34', num_classes=num_classes).cuda()
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)
    steps_per_epoch = len(train_loader) if args.max_steps_per_epoch <= 0 else min(len(train_loader), args.max_steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * steps_per_epoch, 1, 1e-6 / args.learning_rate))

    os.makedirs(args.save, exist_ok=True)
    tag = 'in100_wrn_s{}_vanilla_in100_{}'.format(args.seed, args.my_info)
    csv_path = os.path.join(args.save, tag + '_training_results.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    for epoch in range(args.epochs):
        t0 = time.time()
        net.train()
        loss_avg = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            loss = F.cross_entropy(net(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2
            if step % 100 == 0:
                print('epoch {} step {}/{} loss {:.4f} lr {:.5f}'.format(epoch, step, steps_per_epoch, loss_avg, scheduler.get_last_lr()[0]), flush=True)
            if step >= steps_per_epoch:
                break
        net.eval()
        correct, tloss = 0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                out = net(x)
                tloss += float(F.cross_entropy(out, y))
                correct += out.argmax(1).eq(y).sum().item()
        test_loss = tloss / len(test_loader)
        test_err = 100.0 * (1 - correct / len(test_data))
        ckpt = os.path.join(args.save, tag + '_epoch_{}.pt'.format(epoch))
        torch.save(net.state_dict(), ckpt)
        with open(csv_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (epoch + 1, time.time() - t0, loss_avg, test_loss, test_err))
        print('Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.4f} | Test Error {:.2f} | saved {}'.format(
            epoch + 1, int(time.time() - t0), loss_avg, test_loss, test_err, ckpt), flush=True)


if __name__ == '__main__':
    main()
