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
"""Layers for defining NCSN++.
"""
from typing import Any, Optional, Tuple
# from . import layers
# from . import up_or_down_sampling
# import flax.linen as nn
# import jax
# import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# conv1x1 = layers.ddpm_conv1x1
# conv3x3 = layers.ddpm_conv3x3
# NIN = layers.NIN
# default_init = layers.default_init

def conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """1x1 convolution with DDPM initialization."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, dilation=dilation, bias=bias)

def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """3x3 convolution with DDPM initialization."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias)

class NIN(nn.Module):
    """Network in Network layer."""
    def __init__(self, in_channels, out_channels):
        super(NIN, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        return self.linear(x)

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""
  def __init__(self, embedding_size=256, scale=1.0):
    super(GaussianFourierProjection, self).__init__()
    self.embedding_size = embedding_size
    self.scale = scale
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Combine(nn.Module):
  """Combine information from skip connections."""
  def __init__(self, method='cat'):
    super(Combine, self).__init__()
    self.method = method

  def forward(self, x, y):
    h = conv1x1(x.shape[1], y.shape[1])(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')

class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""
  def __init__(self, skip_rescale=False, init_scale=0.):
    super(AttnBlockpp, self).__init__()
    self.skip_rescale = skip_rescale
    self.init_scale = init_scale

  def forward(self, x):
    B, C, H, W = x.shape
    h = nn.GroupNorm(num_groups=min(C // 4, 32), num_channels=C).to(x.device)(x)
    q = NIN(C, C).to(x.device)(h)
    k = NIN(C, C).to(x.device)(h)
    v = NIN(C, C).to(x.device)(h)

    w = torch.einsum('bchw,bCHW->bhwHW', q, k) * (int(C) ** (-0.5))
    w = w.view(B, H, W, H * W)
    w = torch.softmax(w, dim=-1)
    w = w.view(B, H, W, H, W)
    h = torch.einsum('bhwHW,bCHW->bchw', w, v)
    h = NIN(C, C).to(x.device)(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

class Upsample(nn.Module):
  def __init__(self, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
    super(Upsample, self).__init__()
    self.out_ch = out_ch
    self.with_conv = with_conv
    self.fir = fir
    self.fir_kernel = fir_kernel

  def forward(self, x):
    B, C, H, W = x.shape
    out_ch = self.out_ch if self.out_ch else C
    if not self.fir:
      h = F.interpolate(x, scale_factor=2, mode='nearest')
      if self.with_conv:
        h = conv3x3(C, out_ch).to(x.device)(h)
    else:
      if not self.with_conv:
        h = upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = Conv2d(C, out_ch, kernel=3, up=True, resample_kernel=self.fir_kernel).to(x.device)(x)

    assert h.shape == (B, out_ch, 2 * H, 2 * W)
    return h

class Downsample(nn.Module):
  def __init__(self, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
    super(Downsample, self).__init__()
    self.out_ch = out_ch
    self.with_conv = with_conv
    self.fir = fir
    self.fir_kernel = fir_kernel

  def forward(self, x):
    B, C, H, W = x.shape
    out_ch = self.out_ch if self.out_ch else C
    if not self.fir:
      if self.with_conv:
        x = conv3x3(C, out_ch, stride=2).to(x.device)(x)
      else:
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
    else:
      if not self.with_conv:
        x = downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = Conv2d(C, out_ch, kernel=3, down=True, resample_kernel=self.fir_kernel).to(x.device)(x)

    assert x.shape == (B, out_ch, H // 2, W // 2)
    return x

class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""
  def __init__(self, act, out_ch=None, conv_shortcut=False, dropout=0.1, skip_rescale=False, init_scale=0.):
    super(ResnetBlockDDPMpp, self).__init__()
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut
    self.dropout = dropout
    self.skip_rescale = skip_rescale
    self.init_scale = init_scale

  def forward(self, x, temb=None, train=True):
    B, C, H, W = x.shape
    out_ch = self.out_ch if self.out_ch else C
    h = self.act(nn.GroupNorm(num_groups=min(C // 4, 32), num_channels=C).to(x.device)(x))
    h = conv3x3(C, out_ch).to(x.device)(h)
    if temb is not None:
      h += nn.Linear(temb.shape[1], out_ch).to(x.device)(self.act(temb))[:, :, None, None]

    h = self.act(nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch).to(x.device)(h))
    h = nn.Dropout(p=self.dropout)(h) if train else nn.Dropout(p=self.dropout)(h).eval()
    h = conv3x3(out_ch, out_ch, init_scale=self.init_scale).to(x.device)(h)
    if C != out_ch:
      if self.conv_shortcut:
        x = conv3x3(C, out_ch).to(x.device)(x)
      else:
        x = NIN(C, out_ch).to(x.device)(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

class ResnetBlockBigGANpp(nn.Module):
  """ResBlock adapted from BigGAN."""
  def __init__(self, act, up=False, down=False, out_ch=None, dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1), skip_rescale=True, init_scale=0.):
    super(ResnetBlockBigGANpp, self).__init__()
    self.act = act
    self.up = up
    self.down = down
    self.out_ch = out_ch
    self.dropout = dropout
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.skip_rescale = skip_rescale
    self.init_scale = init_scale

  def forward(self, x, temb=None, train=True):
    B, C, H, W = x.shape
    out_ch = self.out_ch if self.out_ch else C
    h = self.act(nn.GroupNorm(num_groups=min(C // 4, 32), num_channels=C).to(x.device)(x))

    if self.up:
      if self.fir:
        h = upsample_2d(h, self.fir_kernel, factor=2)
        x = upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        x = F.interpolate(x, scale_factor=2, mode='nearest')
    elif self.down:
      if self.fir:
        h = downsample_2d(h, self.fir_kernel, factor=2)
        x = downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = F.avg_pool2d(h, kernel_size=2, stride=2, padding=0)
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

    h = conv3x3(C, out_ch).to(x.device)(h)
    if temb is not None:
      h += nn.Linear(temb.shape[1], out_ch).to(x.device)(self.act(temb))[:, :, None, None]

    h = self.act(nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch).to(x.device)(h))
    h = nn.Dropout(p=self.dropout)(h) if train else nn.Dropout(p=self.dropout)(h).eval()
    h = conv3x3(out_ch, out_ch, init_scale=self.init_scale).to(x.device)(h)
    if C != out_ch or self.up or self.down:
      x = conv1x1(C, out_ch).to(x.device)(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

# Define upsample_2d and downsample_2d functions
def upsample_2d(x, k=None, factor=2, gain=1, data_format='NCHW'):
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor
    return _simple_upfirdn_2d(
        x,
        k,
        up=factor,
        pad0=(p + 1) // 2 + factor - 1,
        pad1=p // 2,
        data_format=data_format
    )

def downsample_2d(x, k=None, factor=2, gain=1, data_format='NCHW'):
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return _simple_upfirdn_2d(
        x,
        k,
        down=factor,
        pad0=(p + 1) // 2,
        pad1=p // 2,
        data_format=data_format
    )

def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k

def _simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0, data_format='NCHW'):
    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)
    y = F.pad(x, (pad0, pad1, pad0, pad1))
    y = F.conv2d(y, k.unsqueeze(0).unsqueeze(0), stride=up, padding=0)
    if down > 1:
        y = F.avg_pool2d(y, down)
    if data_format == 'NHWC':
        y = y.permute(0, 2, 3, 1)
    return y

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, up=False, down=False, resample_kernel=(1, 3, 3, 1)):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
    
    def forward(self, x):
        if self.up:
            x = upsample_2d(x, self.resample_kernel, factor=2)
        elif self.down:
            x = downsample_2d(x, self.resample_kernel, factor=2)
        return self.conv(x)
