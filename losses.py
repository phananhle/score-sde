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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE

def batch_mul(a, b):
    return a * b

def get_optimizer(config, model_params):
    """Returns a PyTorch optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(model_params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')
    return optimizer

def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, grad, lr, warmup=config.optim.warmup, grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            lr = lr * min(params.step / warmup, 1.0)
        
        if grad_clip >= 0:
            # Compute global gradient norm
            grad_norm = torch.sqrt(sum([torch.sum(x ** 2) for x in grad]))
            # Clip gradient
            clipped_grad = [x * grad_clip / max(grad_norm, grad_clip) for x in grad]
        else:
            clipped_grad = grad

        optimizer.zero_grad()
        for p, g in zip(params.parameters(), clipped_grad):
            p.grad = g
        optimizer.step()

    return optimize_fn

def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbitrary SDEs."""
    reduce_op = torch.mean if reduce_mean else lambda x: 0.5 * torch.sum(x)

    def loss_fn(rng, params, states, batch):
        """Compute the loss function."""
        score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)
        data = batch['image']

        t = torch.rand(data.shape[0]) * (sde.T - eps) + eps
        z = torch.randn_like(data)
        mean, std = sde.marginal_prob(data, t)
        perturbed_data = mean + batch_mul(std, z)
        score, new_model_state = score_fn(perturbed_data, t, rng)

        if not likelihood_weighting:
            losses = torch.square(batch_mul(score, std) + z)
            losses = reduce_op(losses.view(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(data), t)[1] ** 2
            losses = torch.square(score + batch_mul(z, 1. / std))
            losses = reduce_op(losses.view(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss, new_model_state

    return loss_fn

def get_smld_loss_fn(vesde, model, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    reduce_op = torch.mean if reduce_mean else lambda x: 0.5 * torch.sum(x)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch['image']
        labels = torch.randint(0, vesde.N, (data.shape[0],))
        sigmas = smld_sigma_array[labels]
        noise = batch_mul(torch.randn_like(data), sigmas)
        perturbed_data = noise + data
        score, new_model_state = model_fn(perturbed_data, labels, rng)
        target = -batch_mul(noise, 1. / (sigmas ** 2))
        losses = torch.square(score - target)
        losses = reduce_op(losses.view(losses.shape[0], -1), dim=-1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss, new_model_state

    return loss_fn

def get_ddpm_loss_fn(vpsde, model, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."
    reduce_op = torch.mean if reduce_mean else lambda x: 0.5 * torch.sum(x)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch['image']
        labels = torch.randint(0, vpsde.N, (data.shape[0],))
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
        noise = torch.randn_like(data)
        perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + batch_mul(sqrt_1m_alphas_cumprod[labels], noise)
        score, new_model_state = model_fn(perturbed_data, labels, rng)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.view(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss, new_model_state

    return loss_fn

def get_step_fn(sde, model, train, optimizer=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function."""
    if continuous:
        loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(rng, state, batch):
        """Running one step of training or evaluation."""
        params = state['params']
        states = state['model_state']
        lr = state['lr']
        ema_rate = state['ema_rate']

        if train:
            optimizer.zero_grad()
            loss, new_model_state = loss_fn(rng, params, states, batch)
            loss.backward()
            optimizer.step()

            new_params_ema = {k: ema_rate * v + (1. - ema_rate) * params[k] for k, v in state['params_ema'].items()}
            step = state['step'] + 1

            new_state = {
                'step': step,
                'optimizer': optimizer,
                'model_state': new_model_state,
                'params_ema': new_params_ema,
                'lr': lr,
                'params': params
            }
        else:
            with torch.no_grad():
                loss, _ = loss_fn(rng, state['params_ema'], states, batch)
            new_state = state

        return new_state, loss.item()

    return step_fn
