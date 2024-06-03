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
"""Common layers for defining score networks.
"""
import functools
import math
import string
from typing import Any, Sequence, Optional

# import flax.linen as nn
# import jax
# import jax.nn as jnn
# import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F

# def get_act(config):
#   """Get activation functions from the config file."""

#   if config.model.nonlinearity.lower() == 'elu':
#     return nn.elu
#   elif config.model.nonlinearity.lower() == 'relu':
#     return nn.relu
#   elif config.model.nonlinearity.lower() == 'lrelu':
#     return functools.partial(nn.leaky_relu, negative_slope=0.2)
#   elif config.model.nonlinearity.lower() == 'swish':
#     return nn.swish
#   else:
#     raise NotImplementedError('activation function does not exist!')

def get_act(config):
    """Get activation functions from the config file."""
    if config.model.nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError('activation function does not exist!')

# def ncsn_conv1x1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
#   """1x1 convolution with PyTorch initialization. Same as NCSNv1/v2."""
#   init_scale = 1e-10 if init_scale == 0 else init_scale
#   kernel_init = jnn.initializers.variance_scaling(1 / 3 * init_scale, 'fan_in',
#                                                   'uniform')
#   kernel_shape = (1, 1) + (x.shape[-1], out_planes)
#   bias_init = lambda key, shape: kernel_init(key, kernel_shape)[0, 0, 0, :]
#   output = nn.Conv(out_planes, kernel_size=(1, 1),
#                    strides=(stride, stride), padding='SAME', use_bias=bias,
#                    kernel_dilation=(dilation, dilation),
#                    kernel_init=kernel_init,
#                    bias_init=bias_init)(x)
#   return output

def ncsn_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """1x1 convolution with PyTorch initialization. Same as NCSNv1/v2."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias, dilation=dilation)

# def default_init(scale=1.):
#   """The same initialization used in DDPM."""
#   scale = 1e-10 if scale == 0 else scale
#   return jnn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

def default_init(scale=1.):
    """The same initialization used in DDPM."""
    return nn.init.kaiming_uniform_

# def ddpm_conv1x1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
#   """1x1 convolution with DDPM initialization."""
#   bias_init = jnn.initializers.zeros
#   output = nn.Conv(out_planes, kernel_size=(1, 1),
#                    strides=(stride, stride), padding='SAME', use_bias=bias,
#                    kernel_dilation=(dilation, dilation),
#                    kernel_init=default_init(init_scale),
#                    bias_init=bias_init)(x)
#   return output

def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """1x1 convolution with DDPM initialization."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias, dilation=dilation)

# def ncsn_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
#   """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
#   init_scale = 1e-10 if init_scale == 0 else init_scale
#   kernel_init = jnn.initializers.variance_scaling(1 / 3 * init_scale, 'fan_in',
#                                                   'uniform')
#   kernel_shape = (3, 3) + (x.shape[-1], out_planes)
#   bias_init = lambda key, shape: kernel_init(key, kernel_shape)[0, 0, 0, :]
#   output = nn.Conv(out_planes,
#                    kernel_size=(3, 3),
#                    strides=(stride, stride),
#                    padding='SAME',
#                    use_bias=bias,
#                    kernel_dilation=(dilation, dilation),
#                    kernel_init=kernel_init,
#                    bias_init=bias_init)(x)
#   return output

def ncsn_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias, dilation=dilation)

# def ddpm_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
#   """3x3 convolution with DDPM initialization."""
#   bias_init = jnn.initializers.zeros
#   output = nn.Conv(
#     out_planes,
#     kernel_size=(3, 3),
#     strides=(stride, stride),
#     padding='SAME',
#     use_bias=bias,
#     kernel_dilation=(dilation, dilation),
#     kernel_init=default_init(init_scale),
#     bias_init=bias_init)(x)
#   return output

def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """3x3 convolution with DDPM initialization."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias, dilation=dilation)

###########################################################################
# Functions below are ported over from the NCSNv1/NCSNv2 codebase:
# https://github.com/ermongroup/ncsn
# https://github.com/ermongroup/ncsnv2
###########################################################################


# class CRPBlock(nn.Module):
#   """CRPBlock for RefineNet. Used in NCSNv2."""
#   features: int
#   n_stages: int
#   act: Any = nn.relu

#   @nn.compact
#   def __call__(self, x):
#     x = self.act(x)
#     path = x
#     for _ in range(self.n_stages):
#       path = nn.max_pool(
#         path, window_shape=(5, 5), strides=(1, 1), padding='SAME')
#       path = ncsn_conv3x3(path, self.features, stride=1, bias=False)
#       x = path + x
#     return x

class CRPBlock(nn.Module):
    """CRPBlock for RefineNet. Used in NCSNv2."""
    def __init__(self, features, n_stages, act=nn.ReLU()):
        super(CRPBlock, self).__init__()
        self.features = features
        self.n_stages = n_stages
        self.act = act
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.convs = nn.ModuleList([ncsn_conv3x3(features, features, stride=1, bias=False) for _ in range(n_stages)])

    def forward(self, x):
        x = self.act(x)
        path = x
        for conv in self.convs:
            path = self.maxpool(path)
            path = conv(path)
            x = path + x
        return x

# class CondCRPBlock(nn.Module):
#   """Noise-conditional CRPBlock for RefineNet. Used in NCSNv1."""
#   features: int
#   n_stages: int
#   normalizer: Any
#   act: Any = nn.relu

#   @nn.compact
#   def __call__(self, x, y):
#     x = self.act(x)
#     path = x
#     for _ in range(self.n_stages):
#       path = self.normalizer()(path, y)
#       path = nn.avg_pool(path, window_shape=(5, 5), strides=(1, 1), padding='SAME')
#       path = ncsn_conv3x3(path, self.features, stride=1, bias=False)
#       x = path + x
#     return x

class CondCRPBlock(nn.Module):
    """Noise-conditional CRPBlock for RefineNet. Used in NCSNv1."""
    def __init__(self, features, n_stages, normalizer, act=nn.ReLU()):
        super(CondCRPBlock, self).__init__()
        self.features = features
        self.n_stages = n_stages
        self.normalizer = normalizer
        self.act = act
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.convs = nn.ModuleList([ncsn_conv3x3(features, features, stride=1, bias=False) for _ in range(n_stages)])

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for conv in self.convs:
            path = self.normalizer()(path, y)
            path = self.avgpool(path)
            path = conv(path)
            x = path + x
        return x

# class RCUBlock(nn.Module):
#   """RCUBlock for RefineNet. Used in NCSNv2."""
#   features: int
#   n_blocks: int
#   n_stages: int
#   act: Any = nn.relu

#   @nn.compact
#   def __call__(self, x):
#     for _ in range(self.n_blocks):
#       residual = x
#       for _ in range(self.n_stages):
#         x = self.act(x)
#         x = ncsn_conv3x3(x, self.features, stride=1, bias=False)
#       x = x + residual
#     return x

class RCUBlock(nn.Module):
    """RCUBlock for RefineNet. Used in NCSNv2."""
    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU()):
        super(RCUBlock, self).__init__()
        self.features = features
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.convs = nn.ModuleList([nn.ModuleList([ncsn_conv3x3(features, features, stride=1, bias=False) for _ in range(n_stages)]) for _ in range(n_blocks)])

    def forward(self, x):
        for conv_block in self.convs:
            residual = x
            for conv in conv_block:
                x = self.act(x)
                x = conv(x)
            x = x + residual
        return x

# class CondRCUBlock(nn.Module):
#   """Noise-conditional RCUBlock for RefineNet. Used in NCSNv1."""
#   features: int
#   n_blocks: int
#   n_stages: int
#   normalizer: Any
#   act: Any = nn.relu

#   @nn.compact
#   def __call__(self, x, y):
#     for _ in range(self.n_blocks):
#       residual = x
#       for _ in range(self.n_stages):
#         x = self.normalizer()(x, y)
#         x = self.act(x)
#         x = ncsn_conv3x3(x, self.features, stride=1, bias=False)
#       x += residual
#     return x

class CondRCUBlock(nn.Module):
    """Noise-conditional RCUBlock for RefineNet. Used in NCSNv1."""
    def __init__(self, features, n_blocks, n_stages, normalizer, act=nn.ReLU()):
        super(CondRCUBlock, self).__init__()
        self.features = features
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.normalizer = normalizer
        self.act = act
        self.convs = nn.ModuleList([nn.ModuleList([ncsn_conv3x3(features, features, stride=1, bias=False) for _ in range(n_stages)]) for _ in range(n_blocks)])

    def forward(self, x, y):
        for conv_block in self.convs:
            residual = x
            for conv in conv_block:
                x = self.normalizer()(x, y)
                x = self.act(x)
                x = conv(x)
            x = x + residual
        return x

# class MSFBlock(nn.Module):
#   """MSFBlock for RefineNet. Used in NCSNv2."""
#   shape: Sequence[int]
#   features: int
#   interpolation: str = 'bilinear'

#   @nn.compact
#   def __call__(self, xs):
#     sums = jnp.zeros((xs[0].shape[0], *self.shape, self.features))
#     for i in range(len(xs)):
#       h = ncsn_conv3x3(xs[i], self.features, stride=1, bias=True)
#       if self.interpolation == 'bilinear':
#         h = jax.image.resize(h, (h.shape[0], *self.shape, h.shape[-1]), 'bilinear')
#       elif self.interpolation == 'nearest_neighbor':
#         h = jax.image.resize(h, (h.shape[0], *self.shape, h.shape[-1]), 'nearest')
#       else:
#         raise ValueError(f'Interpolation {self.interpolation} does not exist!')
#       sums = sums + h
#     return sums

class MSFBlock(nn.Module):
    """MSFBlock for RefineNet. Used in NCSNv2."""
    def __init__(self, shape, features, interpolation='bilinear'):
        super(MSFBlock, self).__init__()
        self.shape = shape
        self.features = features
        self.interpolation = interpolation
        self.convs = nn.ModuleList([ncsn_conv3x3(features, features, stride=1, bias=True) for _ in range(len(xs))])

    def forward(self, xs):
        sums = torch.zeros((xs[0].shape[0], *self.shape, self.features), device=xs[0].device)
        for i, conv in enumerate(self.convs):
            h = conv(xs[i])
            if self.interpolation == 'bilinear':
                h = F.interpolate(h, size=self.shape, mode='bilinear', align_corners=False)
            elif self.interpolation == 'nearest_neighbor':
                h = F.interpolate(h, size=self.shape, mode='nearest')
            else:
                raise ValueError(f'Interpolation {self.interpolation} does not exist!')
            sums = sums + h
        return sums

# class CondMSFBlock(nn.Module):
#   """Noise-conditional MSFBlock for RefineNet. Used in NCSNv1."""
#   shape: Sequence[int]
#   features: int
#   normalizer: Any
#   interpolation: str = 'bilinear'

#   @nn.compact
#   def __call__(self, xs, y):
#     sums = jnp.zeros((xs[0].shape[0], *self.shape, self.features))
#     for i in range(len(xs)):
#       h = self.normalizer()(xs[i], y)
#       h = ncsn_conv3x3(h, self.features, stride=1, bias=True)
#       if self.interpolation == 'bilinear':
#         h = jax.image.resize(h, (h.shape[0], *self.shape, h.shape[-1]), 'bilinear')
#       elif self.interpolation == 'nearest_neighbor':
#         h = jax.image.resize(h, (h.shape[0], *self.shape, h.shape[-1]), 'nearest')
#       else:
#         raise ValueError(f'Interpolation {self.interpolation} does not exist')
#       sums = sums + h
#     return sums

class CondMSFBlock(nn.Module):
    """Noise-conditional MSFBlock for RefineNet. Used in NCSNv1."""
    def __init__(self, shape, features, normalizer, interpolation='bilinear'):
        super(CondMSFBlock, self).__init__()
        self.shape = shape
        self.features = features
        self.normalizer = normalizer
        self.interpolation = interpolation
        self.convs = nn.ModuleList([ncsn_conv3x3(features, features, stride=1, bias=True) for _ in range(len(xs))])

    def forward(self, xs, y):
        sums = torch.zeros((xs[0].shape[0], *self.shape, self.features), device=xs[0].device)
        for i, conv in enumerate(self.convs):
            h = self.normalizer()(xs[i], y)
            h = conv(h)
            if self.interpolation == 'bilinear':
                h = F.interpolate(h, size=self.shape, mode='bilinear', align_corners=False)
            elif self.interpolation == 'nearest_neighbor':
                h = F.interpolate(h, size=self.shape, mode='nearest')
            else:
                raise ValueError(f'Interpolation {self.interpolation} does not exist')
            sums = sums + h
        return sums

# class RefineBlock(nn.Module):
#   """RefineBlock for building NCSNv2 RefineNet."""
#   output_shape: Sequence[int]
#   features: int
#   act: Any = nn.relu
#   interpolation: str = 'bilinear'
#   start: bool = False
#   end: bool = False

#   @nn.compact
#   def __call__(self, xs):
#     rcu_block = functools.partial(RCUBlock, n_blocks=2, n_stages=2, act=self.act)
#     rcu_block_output = functools.partial(RCUBlock,
#                                          features=self.features,
#                                          n_blocks=3 if self.end else 1,
#                                          n_stages=2,
#                                          act=self.act)
#     hs = []
#     for i in range(len(xs)):
#       h = rcu_block(features=xs[i].shape[-1])(xs[i])
#       hs.append(h)

#     if not self.start:
#       msf = functools.partial(MSFBlock, features=self.features, interpolation=self.interpolation)
#       h = msf(shape=self.output_shape)(hs)
#     else:
#       h = hs[0]

#     crp = functools.partial(CRPBlock, features=self.features, n_stages=2, act=self.act)
#     h = crp()(h)
#     h = rcu_block_output()(h)
#     return h

class RefineBlock(nn.Module):
    """RefineBlock for building NCSNv2 RefineNet."""
    def __init__(self, output_shape, features, act=nn.ReLU(), interpolation='bilinear', start=False, end=False):
        super(RefineBlock, self).__init__()
        self.output_shape = output_shape
        self.features = features
        self.act = act
        self.interpolation = interpolation
        self.start = start
        self.end = end
        self.rcu_blocks = nn.ModuleList([RCUBlock(features=features, n_blocks=2, n_stages=2, act=act) for _ in range(len(xs))])
        self.rcu_block_output = RCUBlock(features=features, n_blocks=3 if self.end else 1, n_stages=2, act=act)
        self.msf = MSFBlock(shape=output_shape, features=features, interpolation=interpolation)
        self.crp = CRPBlock(features=features, n_stages=2, act=act)

    def forward(self, xs):
        hs = []
        for i, rcu_block in enumerate(self.rcu_blocks):
            h = rcu_block(xs[i])
            hs.append(h)

        if not self.start:
            h = self.msf(hs)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.rcu_block_output(h)
        return h

# class CondRefineBlock(nn.Module):
#   """Noise-conditional RefineBlock for building NCSNv1 RefineNet."""
#   output_shape: Sequence[int]
#   features: int
#   normalizer: Any
#   act: Any = nn.relu
#   interpolation: str = 'bilinear'
#   start: bool = False
#   end: bool = False

#   @nn.compact
#   def __call__(self, xs, y):
#     rcu_block = functools.partial(CondRCUBlock, n_blocks=2, n_stages=2, act=self.act, normalizer=self.normalizer)
#     rcu_block_output = functools.partial(CondRCUBlock,
#                                          features=self.features,
#                                          n_blocks=3 if self.end else 1,
#                                          n_stages=2, act=self.act,
#                                          normalizer=self.normalizer)
#     hs = []
#     for i in range(len(xs)):
#       h = rcu_block(features=xs[i].shape[-1])(xs[i], y)
#       hs.append(h)

#     if not self.start:
#       msf = functools.partial(CondMSFBlock,
#                               features=self.features,
#                               interpolation=self.interpolation,
#                               normalizer=self.normalizer)
#       h = msf(shape=self.output_shape)(hs, y)
#     else:
#       h = hs[0]

#     crp = functools.partial(CondCRPBlock,
#                             features=self.features,
#                             n_stages=2, act=self.act,
#                             normalizer=self.normalizer)
#     h = crp()(h, y)
#     h = rcu_block_output()(h, y)
#     return h

class CondRefineBlock(nn.Module):
    """Noise-conditional RefineBlock for building NCSNv1 RefineNet."""
    def __init__(self, output_shape, features, normalizer, act=nn.ReLU(), interpolation='bilinear', start=False, end=False):
        super(CondRefineBlock, self).__init__()
        self.output_shape = output_shape
        self.features = features
        self.normalizer = normalizer
        self.act = act
        self.interpolation = interpolation
        self.start = start
        self.end = end
        self.rcu_blocks = nn.ModuleList([CondRCUBlock(features=features, n_blocks=2, n_stages=2, act=act, normalizer=normalizer) for _ in range(len(xs))])
        self.rcu_block_output = CondRCUBlock(features=features, n_blocks=3 if self.end else 1, n_stages=2, act=act, normalizer=normalizer)
        self.msf = CondMSFBlock(shape=output_shape, features=features, interpolation=interpolation, normalizer=normalizer)
        self.crp = CondCRPBlock(features=features, n_stages=2, act=act, normalizer=normalizer)

    def forward(self, xs, y):
        hs = []
        for i, rcu_block in enumerate(self.rcu_blocks):
            h = rcu_block(xs[i], y)
            hs.append(h)

        if not self.start:
            h = self.msf(hs, y)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.rcu_block_output(h, y)
        return h

# class ConvMeanPool(nn.Module):
#   """ConvMeanPool for building the ResNet backbone."""
#   output_dim: int
#   kernel_size: int = 3
#   biases: bool = True

#   @nn.compact
#   def __call__(self, inputs):
#     output = nn.Conv(features=self.output_dim,
#                      kernel_size=(self.kernel_size, self.kernel_size),
#                      strides=(1, 1),
#                      padding='SAME',
#                      use_bias=self.biases)(inputs)
#     output = sum([
#       output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
#       output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]
#     ]) / 4.
#     return output

class ConvMeanPool(nn.Module):
    """ConvMeanPool for building the ResNet backbone."""
    def __init__(self, output_dim, kernel_size=3, biases=True):
        super(ConvMeanPool, self).__init__()
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.biases = biases
        self.conv = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=biases)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4.
        return output

# class MeanPoolConv(nn.Module):
#   """MeanPoolConv for building the ResNet backbone."""
#   output_dim: int
#   kernel_size: int = 3
#   biases: bool = True

#   @nn.compact
#   def __call__(self, inputs):
#     output = inputs
#     output = sum([
#       output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
#       output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]
#     ]) / 4.
#     output = nn.Conv(
#       features=self.output_dim,
#       kernel_size=(self.kernel_size, self.kernel_size),
#       strides=(1, 1),
#       padding='SAME',
#       use_bias=self.biases)(output)
#     return output

class MeanPoolConv(nn.Module):
    """MeanPoolConv for building the ResNet backbone."""
    def __init__(self, output_dim, kernel_size=3, biases=True):
        super(MeanPoolConv, self).__init__()
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.biases = biases
        self.conv = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=biases)

    def forward(self, inputs):
        output = (inputs[:, :, ::2, ::2] + inputs[:, :, 1::2, ::2] + inputs[:, :, ::2, 1::2] + inputs[:, :, 1::2, 1::2]) / 4.
        output = self.conv(output)
        return output

# class ResidualBlock(nn.Module):
#   """The residual block for defining the ResNet backbone. Used in NCSNv2."""
#   output_dim: int
#   normalization: Any
#   resample: Optional[str] = None
#   act: Any = nn.elu
#   dilation: int = 1

#   @nn.compact
#   def __call__(self, x):
#     h = self.normalization()(x)
#     h = self.act(h)
#     if self.resample == 'down':
#       h = ncsn_conv3x3(h, h.shape[-1], dilation=self.dilation)
#       h = self.normalization()(h)
#       h = self.act(h)
#       if self.dilation > 1:
#         h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
#         shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
#       else:
#         h = ConvMeanPool(output_dim=self.output_dim)(h)
#         shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
#     elif self.resample is None:
#       if self.dilation > 1:
#         if self.output_dim == x.shape[-1]:
#           shortcut = x
#         else:
#           shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
#         h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
#         h = self.normalization()(h)
#         h = self.act(h)
#         h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
#       else:
#         if self.output_dim == x.shape[-1]:
#           shortcut = x
#         else:
#           shortcut = ncsn_conv1x1(x, self.output_dim)
#         h = ncsn_conv3x3(h, self.output_dim)
#         h = self.normalization()(h)
#         h = self.act(h)
#         h = ncsn_conv3x3(h, self.output_dim)

#     return h + shortcut

class ResidualBlock(nn.Module):
    """The residual block for defining the ResNet backbone. Used in NCSNv2."""
    def __init__(self, output_dim, normalization, resample=None, act=nn.ELU(), dilation=1):
        super(ResidualBlock, self).__init__()
        self.output_dim = output_dim
        self.normalization = normalization
        self.resample = resample
        self.act = act
        self.dilation = dilation
        self.conv1 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
        self.conv_shortcut = ncsn_conv3x3(output_dim, output_dim, dilation=dilation) if dilation > 1 else ncsn_conv1x1(output_dim, output_dim)

    def forward(self, x):
        h = self.normalization()(x)
        h = self.act(h)
        if self.resample == 'down':
            h = self.conv1(h)
            h = self.normalization()(h)
            h = self.act(h)
            if self.dilation > 1:
                h = self.conv2(h)
                shortcut = self.conv_shortcut(x)
            else:
                h = ConvMeanPool(output_dim=self.output_dim)(h)
                shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
        elif self.resample is None:
            if self.dilation > 1:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = self.conv_shortcut(x)
                h = self.conv1(h)
                h = self.normalization()(h)
                h = self.act(h)
                h = self.conv2(h)
            else:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv1x1(x, self.output_dim)
                h = ncsn_conv3x3(h, self.output_dim)
                h = self.normalization()(h)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim)
        return h + shortcut

# class ConditionalResidualBlock(nn.Module):
#   """The noise-conditional residual block for building NCSNv1."""
#   output_dim: int
#   normalization: Any
#   resample: Optional[str] = None
#   act: Any = nn.elu
#   dilation: int = 1

#   @nn.compact
#   def __call__(self, x, y):
#     h = self.normalization()(x, y)
#     h = self.act(h)
#     if self.resample == 'down':
#       h = ncsn_conv3x3(h, h.shape[-1], dilation=self.dilation)
#       h = self.normalization(h, y)
#       h = self.act(h)
#       if self.dilation > 1:
#         h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
#         shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
#       else:
#         h = ConvMeanPool(output_dim=self.output_dim)(h)
#         shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
#     elif self.resample is None:
#       if self.dilation > 1:
#         if self.output_dim == x.shape[-1]:
#           shortcut = x
#         else:
#           shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
#         h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
#         h = self.normalization()(h, y)
#         h = self.act(h)
#         h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
#       else:
#         if self.output_dim == x.shape[-1]:
#           shortcut = x
#         else:
#           shortcut = ncsn_conv1x1(x, self.output_dim)
#         h = ncsn_conv3x3(h, self.output_dim)
#         h = self.normalization()(h, y)
#         h = self.act(h)
#         h = ncsn_conv3x3(h, self.output_dim)

#     return h + shortcut

class ConditionalResidualBlock(nn.Module):
    """The noise-conditional residual block for building NCSNv1."""
    def __init__(self, output_dim, normalization, resample=None, act=nn.ELU(), dilation=1):
        super(ConditionalResidualBlock, self).__init__()
        self.output_dim = output_dim
        self.normalization = normalization
        self.resample = resample
        self.act = act
        self.dilation = dilation
        self.conv1 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
        self.conv_shortcut = ncsn_conv3x3(output_dim, output_dim, dilation=dilation) if dilation > 1 else ncsn_conv1x1(output_dim, output_dim)

    def forward(self, x, y):
        h = self.normalization()(x, y)
        h = self.act(h)
        if self.resample == 'down':
            h = self.conv1(h)
            h = self.normalization()(h, y)
            h = self.act(h)
            if self.dilation > 1:
                h = self.conv2(h)
                shortcut = self.conv_shortcut(x)
            else:
                h = ConvMeanPool(output_dim=self.output_dim)(h)
                shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
        elif self.resample is None:
            if self.dilation > 1:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = self.conv_shortcut(x)
                h = self.conv1(h)
                h = self.normalization()(h, y)
                h = self.act(h)
                h = self.conv2(h)
            else:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv1x1(x, self.output_dim)
                h = ncsn_conv3x3(h, self.output_dim)
                h = self.normalization()(h, y)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim)
        return h + shortcut

###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################


# def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
#   assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
#   half_dim = embedding_dim // 2
#   # magic number 10000 is from transformers
#   emb = math.log(max_positions) / (half_dim - 1)
#   # emb = math.log(2.) / (half_dim - 1)
#   emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
#   # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
#   # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
#   emb = timesteps[:, None] * emb[None, :]
#   emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
#   if embedding_dim % 2 == 1:  # zero pad
#     emb = jnp.pad(emb, [[0, 0], [0, 1]])
#   assert emb.shape == (timesteps.shape[0], embedding_dim)
#   return emb

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

# class NIN(nn.Module):
#   num_units: int
#   init_scale: float = 0.1

#   @nn.compact
#   def __call__(self, x):
#     in_dim = int(x.shape[-1])
#     W = self.param('W', default_init(scale=self.init_scale), (in_dim, self.num_units))
#     b = self.param('b', jnn.initializers.zeros, (self.num_units,))
#     y = contract_inner(x, W) + b
#     assert y.shape == x.shape[:-1] + (self.num_units,)
#     return y

class NIN(nn.Module):
    def __init__(self, num_units, init_scale=0.1):
        super(NIN, self).__init__()
        self.num_units = num_units
        self.init_scale = init_scale
        self.lin = nn.Linear(in_features=num_units, out_features=num_units)

    def forward(self, x):
        y = self.lin(x)
        assert y.shape == x.shape[:-1] + (self.num_units,)
        return y

# def _einsum(a, b, c, x, y):
#   einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
#   return jnp.einsum(einsum_str, x, y)

# def contract_inner(x, y):
#   """tensordot(x, y, 1)."""
#   x_chars = list(string.ascii_lowercase[:len(x.shape)])
#   y_chars = list(string.ascii_uppercase[:len(y.shape)])
#   assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
#   y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
#   out_chars = x_chars[:-1] + y_chars[1:]
#   return _einsum(x_chars, y_chars, out_chars, x, y)

def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    return torch.tensordot(x, y, dims=1)

# class AttnBlock(nn.Module):
#   """Channel-wise self-attention block."""
#   normalize: Any

#   @nn.compact
#   def __call__(self, x):
#     B, H, W, C = x.shape
#     h = self.normalize()(x)
#     q = NIN(C)(h)
#     k = NIN(C)(h)
#     v = NIN(C)(h)

#     w = jnp.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
#     w = jnp.reshape(w, (B, H, W, H * W))
#     w = jax.nn.softmax(w, axis=-1)
#     w = jnp.reshape(w, (B, H, W, H, W))
#     h = jnp.einsum('bhwHW,bHWc->bhwc', w, v)
#     h = NIN(C, init_scale=0.)(h)
#     return x + h

class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""
    def __init__(self, channels, normalize):
        super(AttnBlock, self).__init__()
        self.channels = channels
        self.normalize = normalize
        self.q = NIN(channels)
        self.k = NIN(channels)
        self.v = NIN(channels)
        self.proj_out = NIN(channels)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.normalize()(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        w = torch.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
        w = w.view(B, H, W, H * W)
        w = F.softmax(w, dim=-1)
        w = w.view(B, H, W, H, W)
        h = torch.einsum('bhwHW,bHWc->bhwc', w, v)
        h = self.proj_out(h)
        return x + h

# class Upsample(nn.Module):
#   with_conv: bool = False

#   @nn.compact
#   def __call__(self, x):
#     B, H, W, C = x.shape
#     h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), 'nearest')
#     if self.with_conv:
#       h = ddpm_conv3x3(h, C)
#     return h

class Upsample(nn.Module):
    def __init__(self, with_conv=False):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = ddpm_conv3x3(self.channels, self.channels)

    def forward(self, x):
        B, H, W, C = x.shape
        h = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            h = self.conv(h)
        return h

# class Downsample(nn.Module):
#   with_conv: bool = False

#   @nn.compact
#   def __call__(self, x):
#     B, H, W, C = x.shape
#     if self.with_conv:
#       x = ddpm_conv3x3(x, C, stride=2)
#     else:
#       x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
#     assert x.shape == (B, H // 2, W // 2, C)
#     return x

class Downsample(nn.Module):
    def __init__(self, with_conv=False):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = ddpm_conv3x3(self.channels, self.channels, stride=2)

    def forward(self, x):
        B, H, W, C = x.shape
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        assert x.shape == (B, H // 2, W // 2, C)
        return x

# class ResnetBlockDDPM(nn.Module):
#   """The ResNet Blocks used in DDPM."""
#   act: Any
#   normalize: Any
#   out_ch: Optional[int] = None
#   conv_shortcut: bool = False
#   dropout: float = 0.5

#   @nn.compact
#   def __call__(self, x, temb=None, train=True):
#     B, H, W, C = x.shape
#     out_ch = self.out_ch if self.out_ch else C
#     h = self.act(self.normalize()(x))
#     h = ddpm_conv3x3(h, out_ch)
#     # Add bias to each feature map conditioned on the time embedding
#     if temb is not None:
#       h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[:, None, None, :]
#     h = self.act(self.normalize()(h))
#     h = nn.Dropout(self.dropout)(h, deterministic=not train)
#     h = ddpm_conv3x3(h, out_ch, init_scale=0.)
#     if C != out_ch:
#       if self.conv_shortcut:
#         x = ddpm_conv3x3(x, out_ch)
#       else:
#         x = NIN(out_ch)(x)
#     return x + h

class ResnetBlockDDPM(nn.Module):
    """The ResNet Blocks used in DDPM."""
    def __init__(self, act, normalize, out_ch=None, conv_shortcut=False, dropout=0.5):
        super(ResnetBlockDDPM, self).__init__()
        self.act = act
        self.normalize = normalize
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.conv1 = ddpm_conv3x3(self.channels, self.channels)
        self.conv2 = ddpm_conv3x3(self.channels, self.channels, init_scale=0.0)
        self.nin = NIN(self.channels)

    def forward(self, x, temb=None, train=True):
        B, H, W, C = x.shape
        out_ch = self.out_ch if self.out_ch else C
        h = self.act(self.normalize()(x))
        h = self.conv1(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += nn.Linear(out_ch, out_ch)(self.act(temb))[:, None, None, :]
        h = self.act(self.normalize()(h))
        h = F.dropout(h, p=self.dropout, training=train)
        h = self.conv2(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = ddpm_conv3x3(x, out_ch)
            else:
                x = self.nin(x)
        return x + h
