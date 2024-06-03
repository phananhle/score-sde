# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for computing FID/Inception scores."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CelebA, LSUN
import os

INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
INCEPTION_DEFAULT_IMAGE_SIZE = 299

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 model returning pool3 and logits."""
    def __init__(self, transform_input=False):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(pretrained=True, transform_input=transform_input)
        self.model.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        x = self.model(x)
        pool3 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return {'pool_3': pool3, 'logits': x}

def get_inception_model():
    model = InceptionV3()
    model.eval()
    return model

def load_dataset_stats(config):
    """Load the pre-computed dataset statistics."""
    if config.data.dataset == 'CIFAR10':
        filename = 'assets/stats/cifar10_stats.npz'
    elif config.data.dataset == 'CELEBA':
        filename = 'assets/stats/celeba_stats.npz'
    elif config.data.dataset == 'LSUN':
        filename = f'assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
    else:
        raise ValueError(f'Dataset {config.data.dataset} stats not found.')

    with open(filename, 'rb') as fin:
        stats = np.load(fin)
        return stats

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Numpy implementation of the Frechet Distance."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def get_activations(dataloader, model, device, num_batches=None):
    """Get activations for the InceptionV3 model."""
    model.to(device)
    model.eval()

    activations = []
    for i, (images, _) in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            activations.append(output['pool_3'].cpu().numpy())

    activations = np.concatenate(activations, axis=0)
    return activations

def calculate_inception_score(dataloader, model, device, num_splits=10):
    """Calculate the Inception Score."""
    model.to(device)
    model.eval()

    preds = []
    for images, _ in dataloader:
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            preds.append(F.softmax(output['logits'], dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    scores = []
    for i in range(num_splits):
        part = preds[(i * preds.shape[0] // num_splits):((i + 1) * preds.shape[0] // num_splits), :]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl_div = np.mean(np.sum(kl_div, 1))
        scores.append(np.exp(kl_div))

    return np.mean(scores), np.std(scores)

def calculate_fid(dataloader, model, real_stats, device, num_batches=None):
    """Calculate the FID score."""
    real_mu, real_sigma = real_stats['mu'], real_stats['sigma']
    activations = get_activations(dataloader, model, device, num_batches=num_batches)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    fid = calculate_frechet_distance(real_mu, real_sigma, mu, sigma)
    return fid

def get_data_loader(dataset_name, batch_size, image_size):
    """Create data loaders for training and evaluation."""
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'CELEBA':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = CelebA(root='./data', split='train', download=True, transform=transform)
    elif dataset_name == 'LSUN':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = LSUN(root='./data', classes=['bedroom_train'], transform=transform)
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not supported.')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# Example usage
if __name__ == "__main__":
    class Config:
        class Data:
            dataset = 'CIFAR10'
            category = 'bedroom'
            image_size = 64

        data = Data()
        training = True

    config = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_inception_model()
    real_stats = load_dataset_stats(config)
    dataloader = get_data_loader(config.data.dataset, batch_size=32, image_size=config.data.image_size)
    
    fid = calculate_fid(dataloader, model, real_stats, device)
    print(f"FID: {fid}")

    inception_score_mean, inception_score_std = calculate_inception_score(dataloader, model, device)
    print(f"Inception Score: {inception_score_mean} ± {inception_score_std}")
