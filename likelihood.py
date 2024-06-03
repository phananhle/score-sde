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

"""Various sampling methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import integrate
from models import utils as mutils

def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        x.requires_grad_(True)
        with torch.enable_grad():
            grad_fn = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(grad_fn, x, create_graph=True)[0]
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn

def get_likelihood_fn(sde, model, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point."""

    def drift_fn(state, x, t):
        """The drift function of the reverse-time SDE."""
        score_fn = mutils.get_score_fn(sde, model, state['params_ema'], state['model_state'], train=False, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def likelihood_fn(prng, pstate, data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim."""

        device = data.device
        shape = data.shape
        if hutchinson_type == 'Gaussian':
            epsilon = torch.randn(shape, device=device)
        elif hutchinson_type == 'Rademacher':
            epsilon = torch.randint(0, 2, shape, device=device).float() * 2 - 1
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

        def ode_func(t, x):
            x_tensor = torch.tensor(x[:-shape[0] * shape[1]], dtype=torch.float32, device=device).view(shape)
            vec_t = torch.ones((x_tensor.shape[0], x_tensor.shape[1]), device=device) * t
            drift = drift_fn(pstate, x_tensor, vec_t).cpu().numpy()
            div_fn = get_div_fn(lambda x, t: drift_fn(pstate, x, t))
            logp_grad = div_fn(x_tensor, vec_t, epsilon).cpu().numpy()
            return np.concatenate([drift, logp_grad], axis=0)

        init = np.concatenate([data.cpu().numpy().flatten(), np.zeros((shape[0] * shape[1],))], axis=0)
        solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev
        zp = torch.tensor(solution.y[:, -1], dtype=torch.float32, device=device)
        z = zp[:-shape[0] * shape[1]].view(shape)
        delta_logp = zp[-shape[0] * shape[1]:].view((shape[0], shape[1]))
        prior_logp = sde.prior_logp(z)
        bpd = -(prior_logp + delta_logp) / np.log(2)
        N = np.prod(shape[2:])
        bpd = bpd / N
        # A hack to convert log-likelihoods to bits/dim
        # based on the gradient of the inverse data normalizer.
        offset = np.log2(torch.autograd.grad(inverse_scaler(torch.tensor(0.)), torch.tensor(0.), retain_graph=True)[0].item()) + 8.
        bpd += offset
        return bpd, z, nfe

    return likelihood_fn
