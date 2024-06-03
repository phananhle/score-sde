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

"""Return training and evaluation/test datasets from config files."""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN, CelebA, LSUN

class CustomDataset(Dataset):
    def __init__(self, tfrecords_path, transform=None):
        import tensorflow as tf
        self.dataset = tf.data.TFRecordDataset(tfrecords_path)
        self.transform = transform

    def __len__(self):
        return len(list(self.dataset))

    def __getitem__(self, idx):
        import tensorflow as tf
        for i, raw_record in enumerate(self.dataset):
            if i == idx:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                image = tf.io.decode_raw(example.features.feature['data'].bytes_list.value[0], tf.uint8)
                image = tf.reshape(image, example.features.feature['shape'].int64_list.value)
                image = tf.transpose(image, (1, 2, 0)).numpy()
                image = image.astype(np.float32) / 255.0
                if self.transform:
                    image = self.transform(image)
                return image, None

def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        return transforms.Normalize((0.5,), (0.5,))
    else:
        return transforms.Lambda(lambda x: x)

def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        return transforms.Lambda(lambda x: (x + 1.) / 2.)
    else:
        return transforms.Lambda(lambda x: x)

def get_transform(config, evaluation=False):
    transform_list = [transforms.Resize((config.data.image_size, config.data.image_size))]
    if config.data.random_flip and not evaluation:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    if not evaluation:
        transform_list.append(get_data_scaler(config))
    return transforms.Compose(transform_list)

def get_dataset(config, additional_dim=None, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.
        additional_dim: An integer or `None`. If present, add one additional dimension to the output data,
        which equals the number of steps jitted together.
        uniform_dequantization: If `True`, add uniform dequantization to images.
        evaluation: If `True`, fix number of epochs to 1.

    Returns:
        train_loader, eval_loader, dataset_builder.
    """
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    transform = get_transform(config, evaluation)

    if config.data.dataset == 'CIFAR10':
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        eval_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif config.data.dataset == 'SVHN':
        train_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
        eval_dataset = SVHN(root='./data', split='test', download=True, transform=transform)
    elif config.data.dataset == 'CELEBA':
        train_dataset = CelebA(root='./data', split='train', download=True, transform=transform)
        eval_dataset = CelebA(root='./data', split='valid', download=True, transform=transform)
    elif config.data.dataset == 'LSUN':
        train_dataset = LSUN(root='./data', classes=['bedroom_train'], transform=transform)
        eval_dataset = LSUN(root='./data', classes=['bedroom_val'], transform=transform)
    elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
        train_dataset = CustomDataset(config.data.tfrecords_path, transform=transform)
        eval_dataset = CustomDataset(config.data.tfrecords_path, transform=transform)
    else:
        raise NotImplementedError(f'Dataset {config.data.dataset} not yet supported.')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, eval_loader
