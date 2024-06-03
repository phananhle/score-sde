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

"""All functions and modules related to model definition.
"""
from typing import Any

# import flax
# import functools
# import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import wideresnet_noise_conditional
# from flax.training import checkpoints
from utils import batch_mul

# The dataclass that stores all training states
# @flax.struct.dataclass
class State:
    def __init__(self, step, optimizer, lr, model_state, ema_rate, params_ema, rng):
        self.step = step
        self.optimizer = optimizer
        self.lr = lr
        self.model_state = model_state
        self.ema_rate = ema_rate
        self.params_ema = params_ema
        self.rng = rng


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
        config: A ConfigDict object parsed from the config file
    Returns:
        sigmas: a numpy array of noise levels
    """
    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max), np.log(config.model.sigma_min),
            config.model.num_scales))

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
        'beta_min': beta_start * (num_diffusion_timesteps - 1),
        'beta_max': beta_end * (num_diffusion_timesteps - 1),
        'num_diffusion_timesteps': num_diffusion_timesteps
    }


def init_model(config):
    """ Initialize a PyTorch model. """
    model_name = config.model.name
    model_def = get_model(model_name)(config=config)
    return model_def, model_def.state_dict(), model_def.state_dict()


def get_model_fn(model, params, states, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: A PyTorch nn.Module object that represents the architecture of score-based model.
        params: A dictionary that contains all trainable parameters.
        states: A dictionary that contains all mutable states.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(x, labels, rng=None):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
                for different models.
            rng: If present, it is the random state for dropout

        Returns:
            A tuple of (model output, new mutable states)
        """
        model.load_state_dict(params)
        if not train:
            model.eval()
            with torch.no_grad():
                output = model(x, labels)
            return output, states
        else:
            model.train()
            output = model(x, labels)
            return output, states

    return model_fn


def get_score_fn(sde, model, params, states, train=False, continuous=False, return_state=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A PyTorch nn.Module object that represents the architecture of the score-based model.
        params: A dictionary that contains all trainable parameters.
        states: A dictionary that contains all other mutable parameters.
        train: `True` for training and `False` for evaluation.
        continuous: If `True`, the score-based model is expected to directly take continuous time steps.
        return_state: If `True`, return the new mutable states alongside the model output.

    Returns:
        A score function.
    """
    model_fn = get_model_fn(model, params, states, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t, rng=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                model_output, state = model_fn(x, labels, rng)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                model_output, state = model_fn(x, labels, rng)
                std = sde.sqrt_1m_alphas_cumprod[labels.int()]

            score = batch_mul(-model_output, 1. / std)
            if return_state:
                return score, state
            else:
                return score

    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t, rng=None):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).int()

            score, state = model_fn(x, labels, rng)
            if return_state:
                return score, state
            else:
                return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def to_flattened_numpy(x):
    """Flatten a PyTorch tensor `x` and convert it to numpy."""
    return x.view(-1).cpu().numpy()


def from_flattened_numpy(x, shape):
    """Form a PyTorch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.tensor(x).view(shape)


def create_classifier(batch_size, ckpt_path):
    """Create a noise-conditional image classifier.

    Args:
        batch_size: The batch size of input data.
        ckpt_path: The path to stored checkpoints for this classifier.

    Returns:
        classifier: A PyTorch nn.Module object that represents the architecture of the classifier.
        classifier_params: A dictionary that contains trainable parameters of the classifier.
    """
    input_shape = (batch_size, 32, 32, 3)
    classifier = wideresnet_noise_conditional.WideResnet(
        blocks_per_group=4,
        channel_multiplier=10,
        num_outputs=10
    )
    classifier.load_state_dict(torch.load(ckpt_path))
    return classifier, classifier.state_dict()


def get_logit_fn(classifier, classifier_params):
    """ Create a logit function for the classifier. """

    def preprocess(data):
        image_mean = torch.tensor([0.49139968, 0.48215841, 0.44653091]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.24703223, 0.24348513, 0.26158784]).view(1, 3, 1, 1)
        return (data - image_mean) / image_std

    def logit_fn(data, ve_noise_scale):
        """Give the logits of the classifier.

        Args:
            data: A PyTorch tensor of the input.
            ve_noise_scale: time conditioning variables in the form of VE SDEs.

        Returns:
            logits: The logits given by the noise-conditional classifier.
        """
        data = preprocess(data)
        classifier.load_state_dict(classifier_params)
        classifier.eval()
        with torch.no_grad():
            logits = classifier(data, ve_noise_scale)
        return logits

    return logit_fn


def get_classifier_grad_fn(logit_fn):
    """Create the gradient function for the classifier in use of class-conditional sampling. """

    def grad_fn(data, ve_noise_scale, labels):
        data.requires_grad = True
        logits = logit_fn(data, ve_noise_scale)
        prob = torch.nn.functional.log_softmax(logits, dim=-1)[torch.arange(labels.shape[0]), labels].sum()
        prob.backward()
        return data.grad

    return grad_fn
