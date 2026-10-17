#!/usr/bin/env python3
import math

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.datasets as dset


class MultiViewCIFAR100(dset.CIFAR100):
    def __init__(self, root, train, transform, num_views=2, download=False):
        super().__init__(root=root, train=train, transform=None, download=download)
        if num_views < 1:
            raise ValueError('num_views must be >= 1')
        self.view_transform = transform
        self.num_views = num_views

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        views = [self.view_transform(image) for _ in range(self.num_views)]
        return torch.stack(views, dim=0), target


class MultiViewImageFolder(dset.ImageFolder):
    def __init__(self, root, transform, num_views=3):
        super().__init__(root, transform=None)
        if num_views < 1:
            raise ValueError('num_views must be >= 1')
        self.view_transform = transform
        self.num_views = num_views

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        views = [self.view_transform(sample) for _ in range(self.num_views)]
        return torch.stack(views, dim=0), target


def compute_class_logits(features, anchor, logit_temperature=0.1):
    if logit_temperature <= 0:
        raise ValueError('logit_temperature must be positive')
    anchor_norm = F.normalize(anchor, dim=-1)
    if features.dim() == 2:
        feature_norm = F.normalize(features, dim=-1)
        return torch.matmul(feature_norm, anchor_norm.t()) / logit_temperature
    if features.dim() == 3:
        feature_norm = F.normalize(features, dim=-1)
        return torch.einsum('bvd,cd->bvc', feature_norm, anchor_norm) / logit_temperature
    raise ValueError('features must be a 2D or 3D tensor')


def compute_robust_logits(view_logits, robust_temperature=1.0, robust_radius=0.0):
    if view_logits.dim() != 3:
        raise ValueError('view_logits must be a 3D tensor [batch, views, classes]')
    if robust_temperature <= 0:
        raise ValueError('robust_temperature must be positive')
    log_num_views = math.log(view_logits.size(1))
    robust_logits = -robust_temperature * (
        torch.logsumexp(-view_logits / robust_temperature, dim=1) - log_num_views
    )
    if robust_radius != 0:
        robust_logits = robust_logits - robust_radius
    return robust_logits


def get_current_robust_weight(epoch, base_weight, start_epoch, ramp_epochs):
    if base_weight <= 0:
        return 0.0
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return float(base_weight)
    progress = min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))
    return float(base_weight) * progress
