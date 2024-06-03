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

# pylint: skip-file
"""Utility code for generating and saving image grids and checkpointing.

   The `save_image` code is copied from
   https://github.com/google/flax/blob/master/examples/vae/utils.py,
   which is a JAX equivalent to the same function in TorchVision
   (https://github.com/pytorch/vision/blob/master/torchvision/utils.py)
"""

import math
from typing import Any, Dict, Optional, TypeVar

# import flax
# import jax
# import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
# import tensorflow as tf

T = TypeVar("T")

def batch_add(a, b):
    return a + b

def batch_mul(a, b):
    return a * b

# def load_training_state(filepath, state):
#   with tf.io.gfile.GFile(filepath, "rb") as f:
#     state = flax.serialization.from_bytes(state, f.read())
#   return state

def save_image(tensor, fp, nrow=8, padding=2, pad_value=0.0, format=None):
    """Make a grid of images and save it into an image file.

    Pixel values are assumed to be within [0, 1].

    Args:
        tensor (array_like): 4D mini-batch images of shape (B x H x W x C).
        fp: A filename(string) or file object.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
        format(Optional):  If omitted, the format to use is determined from the
            filename extension. If a file object was used instead of a filename, this
            parameter should always be used.
    """
    if not (isinstance(tensor, torch.Tensor) or
            (isinstance(tensor, list) and
             all(isinstance(t, torch.Tensor) for t in tensor))):
        raise TypeError("array_like of tensors expected, got {}".format(
            type(tensor)))

    tensor = torch.as_tensor(tensor)

    if tensor.ndimension() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    # Make a grid of images
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = torch.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(2, x * width + padding, width - padding).copy_(tensor[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add(0.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

def flatten_dict(config):
    """Flatten a hierarchical dict to a simple dict."""
    new_dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            sub_dict = flatten_dict(value)
            for subkey, subvalue in sub_dict.items():
                new_dict[key + "/" + subkey] = subvalue
        elif isinstance(value, tuple):
            new_dict[key] = str(value)
        else:
            new_dict[key] = value
    return new_dict

def save_checkpoint(directory, state, step):
    """Save the training state to a checkpoint."""
    torch.save(state, f"{directory}/checkpoint_{step}.pth")

def restore_checkpoint(directory, state, step):
    """Restore the training state from a checkpoint."""
    state = torch.load(f"{directory}/checkpoint_{step}.pth")
    return state
