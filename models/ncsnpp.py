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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np
import math

# Define utility functions
def get_sigmas(config):
    sigmas = np.exp(np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))
    return torch.tensor(sigmas, dtype=torch.float32)

def get_act(config):
    if config.model.nonlinearity == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif config.model.nonlinearity == 'swish':
        return lambda x: x * torch.sigmoid(x)
    else:
        raise NotImplementedError(f"activation {config.model.nonlinearity} not implemented")

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size, scale):
        super(GaussianFourierProjection, self).__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# Define layers
def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias)
    init.xavier_uniform_(conv.weight, gain=init.calculate_gain('relu') * init_scale)
    if bias:
        init.zeros_(conv.bias)
    return conv

def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
    init.xavier_uniform_(conv.weight, gain=init.calculate_gain('relu') * init_scale)
    if bias:
        init.zeros_(conv.bias)
    return conv

class Combine(nn.Module):
    def __init__(self, method):
        super(Combine, self).__init__()
        self.method = method

    def forward(self, x, y):
        if self.method == 'cat':
            return torch.cat([x, y], dim=1)
        elif self.method == 'sum':
            return x + y
        else:
            raise ValueError(f'Method {self.method} not recognized.')

class AttnBlockpp(nn.Module):
    def __init__(self, skip_rescale=False, init_scale=0.):
        super(AttnBlockpp, self).__init__()
        self.skip_rescale = skip_rescale
        self.norm = nn.GroupNorm(num_groups=32, num_channels=None)  # Adjust this as needed
        self.q = nn.Conv2d(None, None, kernel_size=1)  # Adjust in_channels and out_channels as needed
        self.k = nn.Conv2d(None, None, kernel_size=1)
        self.v = nn.Conv2d(None, None, kernel_size=1)
        self.proj_out = nn.Conv2d(None, None, kernel_size=1)
        init.xavier_uniform_(self.proj_out.weight, gain=init_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        w = torch.einsum('bchw,bcHW->bhwHW', q, k) * (C ** -0.5)
        w = w.view(B, H, W, H * W)
        w = F.softmax(w, dim=-1)
        w = w.view(B, H, W, H, W)
        h = torch.einsum('bhwHW,bcHW->bchw', w, v)
        h = self.proj_out(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)

class Upsample(nn.Module):
    def __init__(self, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if self.fir:
            # Implement FIR upsample here
            pass
        else:
            h = F.interpolate(x, scale_factor=2, mode='nearest')
            if self.with_conv:
                h = ddpm_conv3x3(C, self.out_ch if self.out_ch else C)(h)
        return h

class Downsample(nn.Module):
    def __init__(self, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if self.fir:
            # Implement FIR downsample here
            pass
        else:
            if self.with_conv:
                x = ddpm_conv3x3(C, self.out_ch if self.out_ch else C, stride=2)(x)
            else:
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        return x

class ResnetBlockDDPMpp(nn.Module):
    def __init__(self, act, out_ch=None, conv_shortcut=False, dropout=0.1, skip_rescale=False, init_scale=0.):
        super(ResnetBlockDDPMpp, self).__init__()
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.skip_rescale = skip_rescale
        self.init_scale = init_scale

    def forward(self, x, temb, train=True):
        B, C, H, W = x.shape
        out_ch = self.out_ch if self.out_ch else C
        h = self.act(self.group_norm()(x))
        h = ddpm_conv3x3(C, out_ch)(h)
        if temb is not None:
            h += nn.Linear(out_ch, kernel_init=default_initializer())(self.act(temb))[:, None, None, :]

        h = self.act(self.group_norm()(h))
        h = F.dropout(h, p=self.dropout, training=train)
        h = ddpm_conv3x3(out_ch, out_ch, init_scale=self.init_scale)(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = ddpm_conv3x3(C, out_ch)(x)
            else:
                x = nn.Linear(out_ch, kernel_init=default_initializer())(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)

# Main model definition
class NCSNpp(nn.Module):
    """NCSN++ model"""
    def __init__(self, config):
        super(NCSNpp, self).__init__()
        self.config = config
        self.act = get_act(config)
        self.sigmas = get_sigmas(config)

        nf = config.model.nf
        ch_mult = config.model.ch_mult
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        num_resolutions = len(ch_mult)

        conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        skip_rescale = config.model.skip_rescale
        resblock_type = config.model.resblock_type.lower()
        progressive = config.model.progressive.lower()
        progressive_input = config.model.progressive_input.lower()
        embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        combine_method = config.model.progressive_combine.lower()
        self.combiner = functools.partial(Combine, method=combine_method)

        if embedding_type == 'fourier':
            assert config.training.continuous, "Fourier features are only used for continuous training."
            self.fourier_proj = GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale)
        elif embedding_type == 'positional':
            self.get_timestep_embedding = get_timestep_embedding
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            self.temb_dense = nn.Sequential(
                nn.Linear(nf, nf * 4),
                nn.ReLU(),
                nn.Linear(nf * 4, nf * 4)
            )
        else:
            self.temb_dense = None

        self.resblock_type = resblock_type
        self.ResnetBlock = ResnetBlockDDPMpp  # Use ResnetBlockDDPMpp for both cases for now

        self.act = get_act(config)
        self.conv1 = ddpm_conv3x3

    def forward(self, x, time_cond, train=True):
        config = self.config
        act = self.act

        nf = config.model.nf
        ch_mult = config.model.ch_mult
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        num_resolutions = len(ch_mult)

        conditional = config.model.conditional
        embedding_type = config.model.embedding_type.lower()

        if embedding_type == 'fourier':
            used_sigmas = time_cond
            temb = self.fourier_proj(torch.log(used_sigmas))
        elif embedding_type == 'positional':
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = self.get_timestep_embedding(timesteps, nf)

        if conditional:
            temb = self.temb_dense(act(temb))
        else:
            temb = None

        AttnBlock = functools.partial(AttnBlockpp, init_scale=config.model.init_scale, skip_rescale=config.model.skip_rescale)

        Upsample = functools.partial(Upsample, with_conv=config.model.resamp_with_conv, fir=config.model.fir, fir_kernel=config.model.fir_kernel)

        if config.model.progressive == 'output_skip':
            pyramid_upsample = functools.partial(Upsample, fir=config.model.fir, fir_kernel=config.model.fir_kernel, with_conv=False)
        elif config.model.progressive == 'residual':
            pyramid_upsample = functools.partial(Upsample, fir=config.model.fir, fir_kernel=config.model.fir_kernel, with_conv=True)

        Downsample = functools.partial(Downsample, with_conv=config.model.resamp_with_conv, fir=config.model.fir, fir_kernel=config.model.fir_kernel)

        if config.model.progressive_input == 'input_skip':
            pyramid_downsample = functools.partial(Downsample, fir=config.model.fir, fir_kernel=config.model.fir_kernel, with_conv=False)
        elif config.model.progressive_input == 'residual':
            pyramid_downsample = functools.partial(Downsample, fir=config.model.fir, fir_kernel=config.model.fir_kernel, with_conv=True)

        if not config.data.centered:
            x = 2 * x - 1.

        input_pyramid = None
        if config.model.progressive_input != 'none':
            input_pyramid = x

        hs = [self.conv1(x, nf)]
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                h = self.ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
                if h.shape[2] in attn_resolutions:
                    h = AttnBlock()(h)
                hs.append(h)

            if i_level != num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = Downsample()(hs[-1])
                else:
                    h = self.ResnetBlock(down=True)(hs[-1], temb, train)

                if config.model.progressive_input == 'input_skip':
                    input_pyramid = pyramid_downsample()(input_pyramid)
                    h = self.combiner()(input_pyramid, h)

                elif config.model.progressive_input == 'residual':
                    input_pyramid = pyramid_downsample(out_ch=h.shape[1])(input_pyramid)
                    if config.model.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = self.ResnetBlock()(h, temb, train)
        h = AttnBlock()(h)
        h = self.ResnetBlock()(h, temb, train)

        pyramid = None

        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                h = self.ResnetBlock(out_ch=nf * ch_mult[i_level])(torch.cat([h, hs.pop()], dim=1), temb, train)

            if h.shape[2] in attn_resolutions:
                h = AttnBlock()(h)

            if config.model.progressive != 'none':
                if i_level == num_resolutions - 1:
                    if config.model.progressive == 'output_skip':
                        pyramid = ddpm_conv3x3(nf * ch_mult[i_level], x.shape[1])(act(nn.GroupNorm(num_groups=32, num_channels=h.shape[1])(h)))
                    elif config.model.progressive == 'residual':
                        pyramid = ddpm_conv3x3(nf * ch_mult[i_level], h.shape[1])(act(nn.GroupNorm(num_groups=32, num_channels=h.shape[1])(h)))
                else:
                    if config.model.progressive == 'output_skip':
                        pyramid = pyramid_upsample()(pyramid)
                        pyramid = pyramid + ddpm_conv3x3(nf * ch_mult[i_level], x.shape[1])(act(nn.GroupNorm(num_groups=32, num_channels=h.shape[1])(h)))
                    elif config.model.progressive == 'residual':
                        pyramid = pyramid_upsample(out_ch=h.shape[1])(pyramid)
                        if config.model.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = Upsample()(h)
                else:
                    h = self.ResnetBlock(up=True)(h, temb, train)

        assert not hs

        if config.model.progressive == 'output_skip':
            h = pyramid
        else:
            h = act(nn.GroupNorm(num_groups=32, num_channels=h.shape[1])(h))
            h = ddpm_conv3x3(h.shape[1], x.shape[1])(h)

        if config.model.scale_by_sigma:
            used_sigmas = used_sigmas.view(x.shape[0], *([1] * len(x.shape[1:])))
            h = h / used_sigmas

        return h

# Register the model
from models.utils import register_model

register_model(NCSNpp, name='ncsnpp')
